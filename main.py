import cv2
import time
import threading
import concurrent.futures

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
from tracking.track_registry import TrackRegistry
from face_biometry.face_capture import FaceCapture
from face_biometry.face_storage import FaceStorage
from face_biometry.face_reid import FaceReID
from face_biometry.face_snapshot import FaceSnapshotBuffer
from events.event_logger import EventLogger
from events.notification import NotificationDispatcher
from events.alerts import AlertManager
from evidence.evidence_manager import EvidenceManager

MIN_TRACK_AGE_FOR_EVENTS  = 10    # frames mínimos antes de aceitar evento de colisão/soco
LOG_PERF_INTERVAL         = 10.0  # segundos entre logs de performance
TEMP_FACE_CLEANUP_INTERVAL = 120.0 # segundos entre limpezas de faces temporárias expiradas
CAMERA_ID                 = "cam_0"


# ---------------------------------------------------------------------------
# Helpers de snapshot thread-safe para o overlay de renderização
# ---------------------------------------------------------------------------

class _Pt:
    __slots__ = ("x", "y", "z", "visibility", "presence")

    def __init__(self, x, y, z=0.0, visibility=1.0, presence=1.0):
        self.x          = x
        self.y          = y
        self.z          = z
        self.visibility = visibility
        self.presence   = presence


def _freeze_landmarks(landmarks):
    if not landmarks:
        return None
    return [
        _Pt(lm.x, lm.y,
            getattr(lm, "z", 0.0),
            getattr(lm, "visibility", 1.0),
            getattr(lm, "presence", 1.0))
        for lm in landmarks
    ]


def _freeze_tracks(tracks):
    return {
        tid: {
            "box":              list(info.get("box") or []),
            "age":              info.get("age", 0),
            "identity":         info.get("identity"),
            "face_status":      info.get("face_status"),
            "face_confidence":  info.get("face_confidence"),
            "face_box":         info.get("face_box"),
            "face_detected_at": info.get("face_detected_at", 0.0),
            "pose":             _freeze_landmarks(info.get("pose")),
        }
        for tid, info in tracks.items()
    }


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def main():
    sys_logger.info("=== Iniciando Cam-Security ===")

    camera = CameraCapture()
    try:
        camera.start()
    except Exception as e:
        sys_logger.error(f"Erro na inicialização da câmera: {e}")
        return

    id_manager      = IDManager()
    object_tracker  = ObjectTracker(id_manager=id_manager, ttl_seconds=300.0)
    track_registry  = TrackRegistry(id_manager=id_manager)
    person_detector = PersonDetector()
    pose_detector   = MediaPipeDetector()
    pose_estimator  = PoseEstimator(required_consecutive_frames=5)

    face_detector = FaceDetector()
    face_storage  = FaceStorage()
    face_capture  = FaceCapture(face_detector=face_detector)
    face_reid     = FaceReID(face_storage=face_storage)
    face_snapshot = FaceSnapshotBuffer()

    evidence_manager = EvidenceManager(fps=camera.fps)
    evidence_manager.register_camera(CAMERA_ID, fps=camera.fps)

    event_logger            = EventLogger()
    notification_dispatcher = NotificationDispatcher()
    alert_manager = AlertManager(
        event_logger=event_logger,
        notification_dispatcher=notification_dispatcher,
        cooldown_seconds=3.0,
        evidence_manager=evidence_manager,
        camera_id=CAMERA_ID,
        face_storage=face_storage,
        face_snapshot=face_snapshot,
    )

    prev_centroids: dict = {}
    prev_wrists:    dict = {}
    punch_counters: dict = {}

    last_infer_time     = time.time()
    last_infer_seq      = -1
    _last_temp_cleanup  = time.time()
    overlay_lock        = threading.Lock()
    overlay = {
        "frame":     None,
        "tracks":    {},
        "alert_ids": [],
        "hud":       "Pessoas: 0 | Dist: LONGE | Colisao: NAO",
    }

    _perf = {
        "yolo_ms": [], "track_ms": [], "pose_ms": [], "face_ms": [],
        "total_ms": [], "frame_age_ms": [], "last_log": time.time(),
    }

    def _avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    def try_reid(person_crop):
        insights = face_capture.capture_face_insights(person_crop, track_id=-1)
        if insights and insights["embedding"] is not None:
            emb = insights["embedding"]
            # 1. Tenta recuperar track_id original do registry (reentrada entre sessões)
            recovered = track_registry.find_by_embedding(emb)
            if recovered is not None:
                return recovered
            # 2. Fallback: match pelo storage em memória (mesma sessão)
            return face_reid.match_embedding(emb)
        return None

    # -----------------------------------------------------------------------
    # Thread de inferência
    # -----------------------------------------------------------------------
    def inference_loop():
        nonlocal last_infer_time, last_infer_seq, _last_temp_cleanup

        while camera.running:
            frame, seq, captured_at = camera.get_frame()
            if frame is None or seq == last_infer_seq:
                time.sleep(0.002)
                continue
            last_infer_seq = seq

            t_start      = time.perf_counter()
            frame_age_ms = (t_start - captured_at) * 1000.0
            frame_ts_ms  = int(captured_at * 1000)
            now          = time.time()
            delta_t      = max(now - last_infer_time, 1e-3)
            last_infer_time = now

            evidence_manager.push_frame(CAMERA_ID, frame, timestamp=now)

            if (now - _last_temp_cleanup) >= TEMP_FACE_CLEANUP_INTERVAL:
                face_storage.expire_temp_faces()
                face_reid.prune_cache(set())
                _last_temp_cleanup = now

            try:
                h, w = frame.shape[:2]

                # 1. YOLO
                t0 = time.perf_counter()
                detected_persons = person_detector.detect_persons(frame)
                yolo_ms = (time.perf_counter() - t0) * 1000.0

                # 2. Tracking
                t0 = time.perf_counter()
                tracks = object_tracker.update(
                    detected_persons, face_reid_callback=try_reid, frame=frame,
                )
                track_ms = (time.perf_counter() - t0) * 1000.0

                # 3. Manutenção dos pools
                active_ids = set(tracks.keys())
                pose_detector.prune_pool(active_ids)
                pose_detector.pose_hold.prune(active_ids)
                face_capture.prune_cache(active_ids)
                face_reid.prune_cache(active_ids)
                face_snapshot.prune_inactive(active_ids, now=now)
                face_snapshot.run_cleanup_if_due(now=now)
                track_registry.expire_old(now=now)
                track_registry.save()

                alert_track_ids   = []
                collision_detected = False
                min_dist_label    = "LONGE"
                track_ids         = list(tracks.keys())
                has_multiple      = len(track_ids) >= 2

                # 3a. Colisão entre pares
                if has_multiple:
                    for i in range(len(track_ids)):
                        for j in range(i + 1, len(track_ids)):
                            idA, idB   = track_ids[i], track_ids[j]
                            infoA, infoB = tracks[idA], tracks[idB]
                            if (infoA.get("age", 0) < MIN_TRACK_AGE_FOR_EVENTS
                                    or infoB.get("age", 0) < MIN_TRACK_AGE_FOR_EVENTS):
                                continue
                            _, dist_label = calculate_relative_distance(infoA["box"], infoB["box"])
                            if detect_collision(infoA["box"], infoB["box"]):
                                collision_detected = True
                                fired = alert_manager.trigger_alert(
                                    "COLISAO", idA,
                                    risk_score=calculate_risk_score("COLISAO", dist_label, 0.0, True),
                                    description=f"Colisão entre #{idA} e #{idB}",
                                    triggered_at=now,
                                )
                                if fired:
                                    track_registry.set_triggered(idA)
                                    track_registry.set_triggered(idB)
                            if dist_label == "PERTO":
                                min_dist_label = "PERTO"
                            elif dist_label == "MEDIO" and min_dist_label != "PERTO":
                                min_dist_label = "MEDIO"

                pose_ms_total = 0.0
                face_ms_total = 0.0
                
                track_crops = {}
                for track_id, info in tracks.items():
                    crop, crop_box = crop_person(frame, info["box"], pad=True)
                    track_crops[track_id] = (crop, crop_box)

                def process_heavy(tid, crp, c_box):
                    t_pose = time.perf_counter()
                    p_lms = pose_detector.process_for_track(tid, crp, c_box, w, h, frame_ts_ms=frame_ts_ms) if crp.size > 0 else None
                    t_pose = (time.perf_counter() - t_pose) * 1000.0
                    
                    t_face = time.perf_counter()
                    f_ins = face_capture.capture_face_insights(crp, track_id=tid) if crp.size > 0 else None
                    t_face = (time.perf_counter() - t_face) * 1000.0
                    
                    return tid, p_lms, t_pose, f_ins, t_face

                heavy_results = {}
                if track_crops:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(track_crops))) as executor:
                        futures = [executor.submit(process_heavy, tid, crp[0], crp[1]) for tid, crp in track_crops.items()]
                        for f in concurrent.futures.as_completed(futures):
                            res = f.result()
                            heavy_results[res[0]] = res

                for track_id, info in tracks.items():
                    crop, crop_box = track_crops[track_id]
                    box       = info["box"]
                    track_age = info.get("age", 0)
                    
                    if track_id in heavy_results:
                        _, pose_lms, p_ms, insights, f_ms = heavy_results[track_id]
                        pose_ms_total += p_ms
                        face_ms_total += f_ms
                    else:
                        pose_lms, insights = None, None

                    tracks[track_id]["pose"] = pose_lms

                    # 3c. Detecção facial com throttle
                    if crop.size > 0:

                        if insights and insights.get("embedding") is not None:
                            emb          = insights["embedding"]
                            clarity      = insights.get("insights", {}).get("clarity", 0.0)
                            quality_norm = min(1.0, clarity / 300.0)

                            # Atualiza face_box e visibilidade
                            fb = insights.get("box")
                            if fb:
                                tracks[track_id]["face_box"]         = [crop_box[0] + fb[0], crop_box[1] + fb[1], fb[2], fb[3]]
                                tracks[track_id]["face_detected_at"] = now
                                tracks[track_id]["face_status"]      = "detectado"

                            # Buffer facial: foto + embedding com throttle/limite por track.
                            # Só alimenta o registry quando o embedding foi realmente aceito.
                            emb_accepted = face_snapshot.on_face_detected(
                                track_id=track_id,
                                face_img=insights.get("face_img"),
                                embedding=emb,
                                now=now,
                            )
                            if emb_accepted:
                                track_registry.add_embedding(
                                    track_id=track_id,
                                    embedding=emb,
                                    person_id=face_storage.get_identity_for_track(track_id),
                                    now=now,
                                )

                            # Identidade já confirmada: apenas propaga, não re-executa Re-ID
                            known_identity = face_storage.get_identity_for_track(track_id)
                            if known_identity is not None:
                                tracks[track_id]["identity"] = known_identity
                            else:
                                # Sem identidade: salva embedding e tenta Re-ID.
                                # identify_track tem cache interno de 30s — não buscará a cada frame.
                                face_storage.save_embedding(track_id, emb, quality=quality_norm)
                                reid = face_reid.identify_track(track_id, emb)
                                if reid.is_match():
                                    tracks[track_id]["identity"]        = reid.person_id
                                    tracks[track_id]["face_status"]     = reid.confidence.lower()
                                    tracks[track_id]["face_confidence"] = reid.similarity
                                    track_registry.set_person(track_id, reid.person_id)
                                    sys_logger.info(
                                        f"[Face] Track #{track_id} → '{reid.person_id}' "
                                        f"sim={reid.similarity:.3f} {reid.confidence}"
                                    )
                        else:
                            # Rosto não visível agora: limpa face_box após TTL
                            if (now - tracks[track_id].get("face_detected_at", 0.0)) > 0.25:
                                tracks[track_id]["face_box"] = None
                            # Identidade conhecida permanece mesmo sem rosto visível
                            known_identity = face_storage.get_identity_for_track(track_id)
                            if known_identity is not None:
                                tracks[track_id]["identity"] = known_identity

                    # 3d. Análise de pose / eventos
                    curr_centroid = calculate_center(box)
                    body_vel      = calculate_velocity(curr_centroid, prev_centroids.get(track_id), delta_t)
                    prev_centroids[track_id] = curr_centroid

                    if pose_lms and len(pose_lms) >= 17:
                        r_shoulder, r_elbow, r_wrist = pose_lms[12], pose_lms[14], pose_lms[16]
                        l_shoulder, l_elbow, l_wrist = pose_lms[11], pose_lms[13], pose_lms[15]

                        curr_wrist = (r_wrist.x * w, r_wrist.y * h)
                        wrist_vel  = calculate_velocity(curr_wrist, prev_wrists.get(track_id), delta_t)
                        prev_wrists[track_id] = curr_wrist

                        max_angle = max(
                            calculate_arm_angle(r_shoulder, r_elbow, r_wrist),
                            calculate_arm_angle(l_shoulder, l_elbow, l_wrist),
                        )
                        is_near   = (min_dist_label == "PERTO") or collision_detected
                        can_punch = has_multiple and (track_age >= MIN_TRACK_AGE_FOR_EVENTS)

                        if detect_punch(wrist_vel, max_angle, can_punch, is_near):
                            cnt = punch_counters.get(track_id, 0) + 1
                            punch_counters[track_id] = cnt
                            if cnt >= 3:
                                alert_track_ids.append(track_id)
                                object_tracker.set_trigger(track_id)
                                fired = alert_manager.trigger_alert(
                                    "SOCO", track_id,
                                    risk_score=calculate_risk_score(
                                        "SOCO", min_dist_label,
                                        max(body_vel, wrist_vel), collision_detected,
                                    ),
                                    description="Ataque/Soco rápido detectado!",
                                    triggered_at=now,
                                )
                                if fired:
                                    track_registry.set_triggered(track_id)
                        else:
                            punch_counters[track_id] = 0
                            is_forbidden, pose_name = pose_estimator.evaluate(pose_lms, track_id=track_id)
                            if is_forbidden:
                                alert_track_ids.append(track_id)
                                object_tracker.set_trigger(track_id)
                                fired = alert_manager.trigger_alert(
                                    pose_name, track_id,
                                    risk_score=calculate_risk_score(
                                        pose_name, min_dist_label, body_vel, collision_detected,
                                    ),
                                    description=f"Pose proibida: {pose_name}",
                                    triggered_at=now,
                                )
                                if fired:
                                    track_registry.set_triggered(track_id)

                total_ms = (time.perf_counter() - t_start) * 1000.0

                _perf["yolo_ms"].append(yolo_ms)
                _perf["track_ms"].append(track_ms)
                _perf["pose_ms"].append(pose_ms_total)
                _perf["face_ms"].append(face_ms_total)
                _perf["total_ms"].append(total_ms)
                _perf["frame_age_ms"].append(frame_age_ms)

                if (now - _perf["last_log"]) >= LOG_PERF_INTERVAL and _perf["total_ms"]:
                    n = len(_perf["total_ms"])
                    sys_logger.info(
                        f"[Perf/{n}frames] "
                        f"YOLO={_avg(_perf['yolo_ms']):.0f}ms "
                        f"Track={_avg(_perf['track_ms']):.0f}ms "
                        f"Pose={_avg(_perf['pose_ms']):.0f}ms "
                        f"Face={_avg(_perf['face_ms']):.0f}ms "
                        f"Total={_avg(_perf['total_ms']):.0f}ms "
                        f"FrameAge={_avg(_perf['frame_age_ms']):.0f}ms "
                        f"FPS_inf={n / LOG_PERF_INTERVAL:.1f}"
                    )
                    for key in ("yolo_ms", "track_ms", "pose_ms", "face_ms", "total_ms", "frame_age_ms"):
                        _perf[key].clear()
                    _perf["last_log"] = now

                with overlay_lock:
                    overlay["frame"]     = frame
                    overlay["tracks"]    = _freeze_tracks(tracks)
                    overlay["alert_ids"] = list(alert_track_ids)
                    overlay["hud"]       = (
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
    last_frame_id = id(None)

    try:
        while camera.running:
            with overlay_lock:
                render_frame = overlay.get("frame")
                tracks       = overlay["tracks"]
                alert_ids    = overlay["alert_ids"]
                hud_text     = overlay["hud"]

            if render_frame is None or id(render_frame) == last_frame_id:
                if cv2.waitKey(1) == 27:
                    break
                time.sleep(0.002)
                continue

            last_frame_id = id(render_frame)
            annotated = person_detector.draw_annotations(
                render_frame, tracks=tracks, alert_track_ids=alert_ids,
            )
            if annotated is not None:
                cv2.putText(annotated, hud_text, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
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
        track_registry.save(force=True)
        pose_detector.close()
        face_detector.close()
        face_capture.close()
        camera.stop()
        cv2.destroyAllWindows()
        sys_logger.info("=== Cam-Security Finalizado ===")


if __name__ == "__main__":
    main()
