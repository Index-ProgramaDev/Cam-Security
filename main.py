"""
Cam-Security — pipeline principal (v3.1)

Correções em relação à v3.0:
  1. Timestamps reais do frame passados ao MediaPipe Pose (RunningMode.VIDEO).
     Antes: incremento fixo de +33ms por chamada (independente do FPS real).
     Agora: timestamp derivado do clock de captura → modelo usa a escala temporal correta.

  2. Stale face corrigido:
     - face_box é limpo no track quando a detecção facial falha (antes permanecia indefinidamente).
     - face_detected_at controla o TTL de propagação no tracker (FACE_BOX_TTL = 0.25s).
     - Apenas identity é preservada enquanto o track existir; face_box exige detecção recente.

  3. Render thread alinhado ao frame inferido:
     - O overlay agora carrega o frame exato que foi usado pela inferência.
     - Elimina a divergência "inferência usou frame N, render mostra frame N+4".

  4. Métricas de latência agregadas a cada LOG_PERF_INTERVAL segundos:
     YOLO, Tracking, Pose, Face, total, FPS de inferência, idade média do frame.
"""

import cv2
import time
import threading

from utils.logger import sys_logger
from utils.math_utils import (
    calculate_center,
    calculate_relative_distance,
    calculate_velocity,
    detect_collision,
    calculate_arm_angle,
    detect_punch,
    calculate_risk_score,
)
from camera.capture import CameraCapture
from detection.person_detector import PersonDetector, crop_person
from detection.face_detector import FaceDetector
from detection.mediapipe_detector import MediaPipeDetector
from detection.pose_estimation import PoseEstimator
from tracking.id_manager import IDManager
from tracking.object_tracker import ObjectTracker
from face_biometry.face_capture import FaceCapture
from face_biometry.face_storage import FaceStorage
from face_biometry.face_reid import FaceReID
from events.event_logger import EventLogger
from events.notification import NotificationDispatcher
from events.alerts import AlertManager

# Mínimo de frames para aceitar eventos de colisão/soco
MIN_TRACK_AGE_FOR_EVENTS = 10

# Intervalo de log de performance (segundos)
LOG_PERF_INTERVAL = 10.0


# ---------------------------------------------------------------------------
# Helpers de serialização (frozen objects para o overlay thread-safe)
# ---------------------------------------------------------------------------

class _Pt:
    __slots__ = ("x", "y", "z", "visibility", "presence")

    def __init__(self, x, y, z=0.0, visibility=1.0, presence=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility
        self.presence = presence


def _freeze_landmarks(landmarks):
    """Copia uma lista de landmarks para objetos simples thread-safe."""
    if not landmarks:
        return None
    return [
        _Pt(lm.x, lm.y, getattr(lm, "z", 0.0),
            getattr(lm, "visibility", 1.0),
            getattr(lm, "presence", 1.0))
        for lm in landmarks
    ]


def _freeze_tracks_snapshot(tracks):
    """
    Cria um snapshot dos tracks para o overlay de renderização.
    Inclui a pose de cada track (já mapeada para o frame) e dados faciais.
    """
    snap = {}
    for tid, info in tracks.items():
        snap[tid] = {
            "box":             list(info.get("box") or []),
            "age":             info.get("age", 0),
            "identity":        info.get("identity"),
            "face_status":     info.get("face_status"),
            "face_confidence": info.get("face_confidence"),
            "face_box":        info.get("face_box"),
            "face_detected_at": info.get("face_detected_at", 0.0),
            "pose":            _freeze_landmarks(info.get("pose")),
        }
    return snap


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sys_logger.info("=== Iniciando Cam-Security (v3.0 — pipeline pose por crop) ===")

    # --- Câmera ---
    camera = CameraCapture()
    try:
        camera.start()
    except Exception as e:
        sys_logger.error(f"Erro na inicialização da câmera: {e}")
        return

    # --- Rastreamento ---
    id_manager = IDManager()
    object_tracker = ObjectTracker(id_manager=id_manager, ttl_seconds=300.0)

    # --- Detecção ---
    person_detector = PersonDetector()
    pose_detector = MediaPipeDetector()
    pose_estimator = PoseEstimator(required_consecutive_frames=5)

    # --- Biometria facial (FaceDetector compartilhado) ---
    face_detector = FaceDetector()                                  # instância única
    face_storage = FaceStorage()
    face_capture = FaceCapture(face_detector=face_detector)         # reutiliza o mesmo
    face_reid = FaceReID(face_storage=face_storage)

    # --- Eventos ---
    event_logger = EventLogger()
    notification_dispatcher = NotificationDispatcher()
    alert_manager = AlertManager(
        event_logger=event_logger,
        notification_dispatcher=notification_dispatcher,
        cooldown_seconds=3.0,
    )

    # --- Estado inter-frame ---
    prev_centroids: dict = {}
    prev_wrists: dict = {}
    punch_counters: dict = {}

    last_infer_time = time.time()
    last_infer_seq = -1
    overlay_lock = threading.Lock()
    overlay = {
        "frame":     None,   # frame exato que foi inferido (para o render thread)
        "tracks":    {},
        "alert_ids": [],
        "hud":       "Pessoas: 0 | Dist: LONGE | Colisao: NAO",
    }

    # Acumuladores de latência para log periódico
    _perf = {
        "yolo_ms": [], "track_ms": [], "pose_ms": [], "face_ms": [],
        "total_ms": [], "frame_age_ms": [], "last_log": time.time(),
    }

    # --- Callback ReID para o tracker ---
    def try_reid(person_crop):
        insights = face_capture.capture_face_insights(person_crop, track_id=-1)
        if insights and insights["embedding"] is not None:
            return face_reid.match_embedding(insights["embedding"])
        return None

    # -----------------------------------------------------------------------
    # Thread de inferência
    # -----------------------------------------------------------------------
    def inference_loop():
        nonlocal last_infer_time, last_infer_seq

        while camera.running:
            frame, seq, captured_at = camera.get_frame()
            if frame is None or seq == last_infer_seq:
                time.sleep(0.002)
                continue
            last_infer_seq = seq

            t_frame_start = time.perf_counter()

            # Idade do frame: tempo desde a captura até agora (ms)
            frame_age_ms = (t_frame_start - captured_at) * 1000.0

            # Timestamp absoluto do frame em ms (para o MediaPipe RunningMode.VIDEO).
            # Derivado de perf_counter para máxima resolução e monotonicidade.
            frame_ts_ms = int(captured_at * 1000)

            now = time.time()
            delta_t = max(now - last_infer_time, 1e-3)
            last_infer_time = now

            try:
                h, w = frame.shape[:2]

                # ── 1. Detecção YOLO ──────────────────────────────────────
                t0 = time.perf_counter()
                detected_persons = person_detector.detect_persons(frame)
                yolo_ms = (time.perf_counter() - t0) * 1000.0

                # ── 2. Tracking ───────────────────────────────────────────
                t0 = time.perf_counter()
                tracks = object_tracker.update(
                    detected_persons,
                    face_reid_callback=try_reid,
                    frame=frame,
                )
                track_ms = (time.perf_counter() - t0) * 1000.0

                # ── 3. Pose + Face por pessoa (crop individual) ───────────
                active_ids = set(tracks.keys())
                pose_detector.prune_pool(active_ids)
                pose_detector.pose_hold.prune(active_ids)
                face_capture.prune_cache(active_ids)

                alert_track_ids = []
                collision_detected = False
                min_dist_label = "LONGE"

                track_ids = list(tracks.keys())
                has_multiple = len(track_ids) >= 2

                # Distância/colisão entre pares
                if has_multiple:
                    for i in range(len(track_ids)):
                        for j in range(i + 1, len(track_ids)):
                            idA, idB = track_ids[i], track_ids[j]
                            infoA, infoB = tracks[idA], tracks[idB]
                            both_ok = (
                                infoA.get("age", 0) >= MIN_TRACK_AGE_FOR_EVENTS
                                and infoB.get("age", 0) >= MIN_TRACK_AGE_FOR_EVENTS
                            )
                            if not both_ok:
                                continue
                            boxA, boxB = infoA["box"], infoB["box"]
                            _, dist_label = calculate_relative_distance(boxA, boxB)
                            is_col = detect_collision(boxA, boxB)
                            if is_col:
                                collision_detected = True
                                risk = calculate_risk_score("COLISAO", dist_label, 0.0, True)
                                alert_manager.trigger_alert(
                                    "COLISAO", idA,
                                    risk_score=risk,
                                    description=f"Colisão entre #{idA} e #{idB}",
                                )
                            if dist_label == "PERTO":
                                min_dist_label = "PERTO"
                            elif dist_label == "MEDIO" and min_dist_label != "PERTO":
                                min_dist_label = "MEDIO"

                pose_ms_total = 0.0
                face_ms_total = 0.0

                for track_id, info in tracks.items():
                    box = info["box"]
                    x1, y1, x2, y2 = map(int, box)
                    track_age = info.get("age", 0)

                    # ── 3a. Crop com padding ───────────────────────────────
                    crop, crop_box = crop_person(frame, box, pad=True)

                    # ── 3b. Pose por crop (mapeada para o frame) ───────────
                    pose_lms = None
                    if crop.size > 0:
                        t0 = time.perf_counter()
                        pose_lms = pose_detector.process_for_track(
                            track_id, crop, crop_box, w, h,
                            frame_ts_ms=frame_ts_ms,
                        )
                        pose_ms_total += (time.perf_counter() - t0) * 1000.0

                    # Armazena pose no track (acessível pelo overlay)
                    tracks[track_id]["pose"] = pose_lms

                    # ── 3c. Detecção facial com throttle ──────────────────
                    if crop.size > 0:
                        t0 = time.perf_counter()
                        insights = face_capture.capture_face_insights(crop, track_id=track_id)
                        face_ms_total += (time.perf_counter() - t0) * 1000.0

                        if insights and insights.get("embedding") is not None:
                            face_storage.save_embedding(track_id, insights["embedding"])
                            fb = insights.get("box")
                            if fb:
                                fx_abs = crop_box[0] + fb[0]
                                fy_abs = crop_box[1] + fb[1]
                                tracks[track_id]["face_box"] = [fx_abs, fy_abs, fb[2], fb[3]]
                                tracks[track_id]["face_detected_at"] = now
                                tracks[track_id]["face_status"] = "detectado"
                        else:
                            # Detecção falhou neste frame: limpa face_box imediatamente.
                            # O tracker propaga por no máximo FACE_BOX_TTL (0.25s);
                            # aqui garantimos que o campo não carregue um resultado stale
                            # quando o FaceCapture retornou resultado atual (não de cache).
                            # Nota: o cache do FaceCapture pode retornar None (throttle expirado
                            # e rosto não encontrado) — esse é o caso que precisamos limpar.
                            face_detected_at = tracks[track_id].get("face_detected_at", 0.0)
                            if (now - face_detected_at) > 0.25:
                                tracks[track_id]["face_box"] = None

                    # ── 3d. Análise de pose / eventos ─────────────────────
                    curr_centroid = calculate_center(box)
                    prev_centroid = prev_centroids.get(track_id)
                    body_vel = calculate_velocity(curr_centroid, prev_centroid, delta_t)
                    prev_centroids[track_id] = curr_centroid

                    if pose_lms and len(pose_lms) >= 17:
                        r_shoulder = pose_lms[12]
                        r_elbow    = pose_lms[14]
                        r_wrist    = pose_lms[16]
                        l_shoulder = pose_lms[11]
                        l_elbow    = pose_lms[13]
                        l_wrist    = pose_lms[15]

                        curr_wrist_pos = (r_wrist.x * w, r_wrist.y * h)
                        prev_wrist_pos = prev_wrists.get(track_id)
                        wrist_vel = calculate_velocity(curr_wrist_pos, prev_wrist_pos, delta_t)
                        prev_wrists[track_id] = curr_wrist_pos

                        r_angle = calculate_arm_angle(r_shoulder, r_elbow, r_wrist)
                        l_angle = calculate_arm_angle(l_shoulder, l_elbow, l_wrist)
                        max_angle = max(r_angle, l_angle)

                        is_near = (min_dist_label == "PERTO") or collision_detected
                        can_punch = has_multiple and (track_age >= MIN_TRACK_AGE_FOR_EVENTS)
                        raw_punch = detect_punch(wrist_vel, max_angle, can_punch, is_near)

                        if raw_punch:
                            cnt = punch_counters.get(track_id, 0) + 1
                            punch_counters[track_id] = cnt
                            if cnt >= 3:
                                alert_track_ids.append(track_id)
                                object_tracker.set_trigger(track_id)
                                risk = calculate_risk_score(
                                    "SOCO", min_dist_label,
                                    max(body_vel, wrist_vel),
                                    collision_detected,
                                )
                                alert_manager.trigger_alert(
                                    "SOCO", track_id,
                                    risk_score=risk,
                                    description="Ataque/Soco rápido detectado!",
                                )
                        else:
                            punch_counters[track_id] = 0
                            is_forbidden, pose_name = pose_estimator.evaluate(
                                pose_lms, track_id=track_id
                            )
                            if is_forbidden:
                                alert_track_ids.append(track_id)
                                object_tracker.set_trigger(track_id)
                                risk = calculate_risk_score(
                                    pose_name, min_dist_label, body_vel, collision_detected
                                )
                                alert_manager.trigger_alert(
                                    pose_name, track_id,
                                    risk_score=risk,
                                    description=f"Pose proibida: {pose_name}",
                                )

                total_ms = (time.perf_counter() - t_frame_start) * 1000.0

                # Acumula métricas
                _perf["yolo_ms"].append(yolo_ms)
                _perf["track_ms"].append(track_ms)
                _perf["pose_ms"].append(pose_ms_total)
                _perf["face_ms"].append(face_ms_total)
                _perf["total_ms"].append(total_ms)
                _perf["frame_age_ms"].append(frame_age_ms)

                # Log periódico de performance
                if (now - _perf["last_log"]) >= LOG_PERF_INTERVAL and _perf["total_ms"]:
                    n = len(_perf["total_ms"])
                    avg = lambda lst: sum(lst) / len(lst) if lst else 0.0
                    fps_inf = n / LOG_PERF_INTERVAL
                    sys_logger.info(
                        f"[Perf/{n}frames] "
                        f"YOLO={avg(_perf['yolo_ms']):.0f}ms "
                        f"Track={avg(_perf['track_ms']):.0f}ms "
                        f"Pose={avg(_perf['pose_ms']):.0f}ms "
                        f"Face={avg(_perf['face_ms']):.0f}ms "
                        f"Total={avg(_perf['total_ms']):.0f}ms "
                        f"FrameAge={avg(_perf['frame_age_ms']):.0f}ms "
                        f"FPS_inf={fps_inf:.1f}"
                    )
                    for key in ("yolo_ms", "track_ms", "pose_ms", "face_ms", "total_ms", "frame_age_ms"):
                        _perf[key].clear()
                    _perf["last_log"] = now

                # ── 4. Publica overlay para o thread de renderização ──────
                with overlay_lock:
                    overlay["frame"]     = frame          # frame exato usado na inferência
                    overlay["tracks"]    = _freeze_tracks_snapshot(tracks)
                    overlay["alert_ids"] = list(alert_track_ids)
                    overlay["hud"] = (
                        f"Pessoas: {len(tracks)} | "
                        f"Dist: {min_dist_label} | "
                        f"Colisao: {'SIM' if collision_detected else 'NAO'}"
                    )

            except Exception as e:
                sys_logger.error(f"[Inference] Erro: {e}")
                import traceback
                traceback.print_exc()

    # -----------------------------------------------------------------------
    # Thread de renderização (thread principal)
    # -----------------------------------------------------------------------
    infer_thread = threading.Thread(target=inference_loop, daemon=True)
    infer_thread.start()
    last_rendered_frame_id = id(None)  # id do objeto frame já exibido

    try:
        while camera.running:
            with overlay_lock:
                render_frame = overlay.get("frame")
                tracks       = overlay["tracks"]
                alert_ids    = overlay["alert_ids"]
                hud_text     = overlay["hud"]

            # Só renderiza quando a inferência produziu um novo resultado
            if render_frame is None or id(render_frame) == last_rendered_frame_id:
                if cv2.waitKey(1) == 27:
                    break
                time.sleep(0.002)
                continue

            last_rendered_frame_id = id(render_frame)

            annotated = person_detector.draw_annotations(
                render_frame,
                tracks=tracks,
                alert_track_ids=alert_ids,
            )
            if annotated is not None:
                cv2.putText(
                    annotated, hud_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                )
                cv2.imshow("Cam-Security | Visão Real", annotated)

            if cv2.waitKey(1) == 27:
                sys_logger.info("Encerrando a pedido do usuário (ESC).")
                break

    except KeyboardInterrupt:
        sys_logger.info("Interrompido via KeyboardInterrupt.")
    except Exception as e:
        sys_logger.error(f"Erro no loop principal: {e}")
        import traceback
        traceback.print_exc()
    finally:
        camera.running = False
        infer_thread.join(timeout=2.0)
        pose_detector.close()
        face_detector.close()
        face_capture.close()
        camera.stop()
        cv2.destroyAllWindows()
        sys_logger.info("=== Cam-Security Finalizado ===")


if __name__ == "__main__":
    main()
