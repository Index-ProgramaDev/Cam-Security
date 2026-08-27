import cv2
import time
import sys
import threading
from utils.logger import sys_logger
from utils.math_utils import (
    calculate_center,
    calculate_relative_distance,
    calculate_velocity,
    detect_collision,
    calculate_arm_angle,
    detect_punch,
    calculate_risk_score
)
from camera.capture import CameraCapture
from detection.person_detector import PersonDetector
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

MIN_TRACK_AGE_FOR_EVENTS = 10  # Mínimo de 10 frames de histórico contínuo para cada track em COLISAO e SOCO


class _Pt:
    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


def _freeze_landmarks(landmarks_list):
    frozen = []
    for landmarks in landmarks_list or []:
        frozen.append([_Pt(lm.x, lm.y, getattr(lm, "z", 0.0)) for lm in landmarks])
    return frozen


def _freeze_faces(faces_data):
    frozen = []
    for face in faces_data or []:
        item = {"box": list(face.get("box") or [])}
        lms = face.get("landmarks")
        if lms:
            item["landmarks"] = [_Pt(lm.x, lm.y, getattr(lm, "z", 0.0)) for lm in lms]
        frozen.append(item)
    return frozen

def main():
    sys_logger.info("=== Iniciando Backend Cam-Security (v2.3 - NMS & Estabilidade Estrita) ===")

    # 1. Câmera Real
    camera = CameraCapture()
    try:
        camera.start()
    except Exception as e:
        sys_logger.error(f"Erro na inicialização da câmera: {e}")
        return

    # 2. Rastreamento e IDs
    id_manager = IDManager()
    object_tracker = ObjectTracker(id_manager=id_manager, ttl_seconds=300.0)

    # 3. Biometria Facial
    face_detector = FaceDetector()
    face_storage = FaceStorage()
    face_capture = FaceCapture()
    face_reid = FaceReID(face_storage=face_storage)

    # 4. Detectores e Pose (Buffer de confirmação de 5 frames)
    person_detector = PersonDetector()
    pose_detector = MediaPipeDetector()
    pose_estimator = PoseEstimator(required_consecutive_frames=5)

    # 5. Eventos e Alertas (Debounce ativo de 3 segundos)
    event_logger = EventLogger()
    notification_dispatcher = NotificationDispatcher()
    alert_manager = AlertManager(event_logger=event_logger, notification_dispatcher=notification_dispatcher, cooldown_seconds=3.0)

    # Histórico de posições
    prev_centroids = {}
    prev_wrists = {}
    punch_counters = {}
    last_infer_time = time.time()
    last_infer_seq = -1
    overlay_lock = threading.Lock()
    overlay = {
        "tracks": {},
        "landmarks": [],
        "faces": [],
        "alert_ids": [],
        "hud": "Pessoas: 0 | Dist: LONGE | Colisao: NAO",
    }

    def try_reid(person_crop):
        insights = face_capture.capture_face_insights(person_crop)
        if insights and insights["embedding"] is not None:
            return face_reid.match_embedding(insights["embedding"])
        return None

    def inference_loop():
        nonlocal last_infer_time, last_infer_seq
        while camera.running:
            frame, seq = camera.get_frame()
            if frame is None or seq == last_infer_seq:
                time.sleep(0.002)
                continue
            last_infer_seq = seq

            now = time.time()
            delta_t = max(now - last_infer_time, 1e-3)
            last_infer_time = now

            try:
                h, w = frame.shape[:2]
                pose_result = pose_detector.process(frame)
                landmarks_list = pose_result.get("pose_landmarks", [])
                detected_persons = person_detector.detect_persons(frame, pose_landmarks_list=landmarks_list)
                tracks = object_tracker.update(detected_persons, face_reid_callback=try_reid, frame=frame)
                faces_data = face_detector.detect_faces(frame)

                track_ids = list(tracks.keys())
                has_multiple_people = len(track_ids) >= 2
                collision_detected = False
                min_dist_label = "LONGE"

                if has_multiple_people:
                    for i in range(len(track_ids)):
                        for j in range(i + 1, len(track_ids)):
                            idA, idB = track_ids[i], track_ids[j]
                            infoA, infoB = tracks[idA], tracks[idB]
                            both_established = (infoA.get("age", 0) >= MIN_TRACK_AGE_FOR_EVENTS) and (infoB.get("age", 0) >= MIN_TRACK_AGE_FOR_EVENTS)
                            if not both_established:
                                continue

                            boxA, boxB = infoA["box"], infoB["box"]
                            _rel_dist, dist_label = calculate_relative_distance(boxA, boxB)
                            is_colliding = detect_collision(boxA, boxB)

                            if is_colliding:
                                collision_detected = True
                                risk = calculate_risk_score("COLISAO", dist_label, 0.0, True)
                                alert_manager.trigger_alert("COLISAO", idA, risk_score=risk, description=f"Colisão entre #{idA} e #{idB}")

                            if dist_label == "PERTO":
                                min_dist_label = "PERTO"
                            elif dist_label == "MEDIO" and min_dist_label != "PERTO":
                                min_dist_label = "MEDIO"

                alert_track_ids = []
                for idx, (track_id, info) in enumerate(tracks.items()):
                    box = info["box"]
                    track_age = info.get("age", 0)
                    curr_centroid = calculate_center(box)
                    prev_centroid = prev_centroids.get(track_id)
                    body_vel = calculate_velocity(curr_centroid, prev_centroid, delta_t)
                    prev_centroids[track_id] = curr_centroid

                    x1, y1, x2, y2 = map(int, box)
                    crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                    if crop.size > 0:
                        insights = face_capture.capture_face_insights(crop)
                        if insights and insights["embedding"] is not None:
                            face_storage.save_embedding(track_id, insights["embedding"])

                    if landmarks_list and idx < len(landmarks_list):
                        landmarks = landmarks_list[idx]
                        r_shoulder, r_elbow, r_wrist = landmarks[12], landmarks[14], landmarks[16]
                        l_shoulder, l_elbow, l_wrist = landmarks[11], landmarks[13], landmarks[15]
                        curr_wrist_pos = (r_wrist.x * w, r_wrist.y * h)
                        prev_wrist_pos = prev_wrists.get(track_id)
                        wrist_vel = calculate_velocity(curr_wrist_pos, prev_wrist_pos, delta_t)
                        prev_wrists[track_id] = curr_wrist_pos
                        r_angle = calculate_arm_angle(r_shoulder, r_elbow, r_wrist)
                        l_angle = calculate_arm_angle(l_shoulder, l_elbow, l_wrist)
                        max_angle = max(r_angle, l_angle)
                        is_near_or_collision = (min_dist_label == "PERTO") or collision_detected
                        can_punch = has_multiple_people and (track_age >= MIN_TRACK_AGE_FOR_EVENTS)
                        raw_punch = detect_punch(wrist_vel, max_angle, has_multiple_people=can_punch, is_near_or_collision=is_near_or_collision)

                        if raw_punch:
                            punch_cnt = punch_counters.get(track_id, 0) + 1
                            punch_counters[track_id] = punch_cnt
                            if punch_cnt >= 3:
                                alert_track_ids.append(track_id)
                                object_tracker.set_trigger(track_id)
                                risk = calculate_risk_score("SOCO", min_dist_label, max(body_vel, wrist_vel), collision_detected)
                                alert_manager.trigger_alert("SOCO", track_id, risk_score=risk, description="Ataque/Soco rápido detectado!")
                        else:
                            punch_counters[track_id] = 0
                            is_forbidden, pose_name = pose_estimator.evaluate(landmarks, track_id=track_id)
                            if is_forbidden:
                                alert_track_ids.append(track_id)
                                object_tracker.set_trigger(track_id)
                                risk = calculate_risk_score(pose_name, min_dist_label, body_vel, collision_detected)
                                alert_manager.trigger_alert(pose_name, track_id, risk_score=risk, description=f"Pose proibida: {pose_name}")

                tracks_snap = {
                    tid: {"box": list(info.get("box", [])), "age": info.get("age", 0)}
                    for tid, info in tracks.items()
                }
                with overlay_lock:
                    overlay["tracks"] = tracks_snap
                    overlay["landmarks"] = _freeze_landmarks(landmarks_list)
                    overlay["faces"] = _freeze_faces(faces_data)
                    overlay["alert_ids"] = list(alert_track_ids)
                    overlay["hud"] = (
                        f"Pessoas: {len(tracks)} | Dist: {min_dist_label} | "
                        f"Colisao: {'SIM' if collision_detected else 'NAO'}"
                    )
            except Exception as e:
                sys_logger.error(f"Erro na inferência: {e}")

    infer_thread = threading.Thread(target=inference_loop, daemon=True)
    infer_thread.start()
    last_shown_seq = -1

    try:
        while camera.running:
            if camera.peek_seq() == last_shown_seq:
                if cv2.waitKey(1) == 27:
                    sys_logger.info("Encerrando a pedido do usuário.")
                    break
                continue

            frame, seq = camera.get_frame()
            if frame is None:
                if cv2.waitKey(1) == 27:
                    sys_logger.info("Encerrando a pedido do usuário.")
                    break
                time.sleep(0.005)
                continue

            last_shown_seq = seq
            with overlay_lock:
                tracks = overlay["tracks"]
                landmarks_list = overlay["landmarks"]
                faces_data = overlay["faces"]
                alert_track_ids = overlay["alert_ids"]
                hud_text = overlay["hud"]

            annotated_frame = person_detector.draw_annotations(
                frame,
                tracks=tracks,
                pose_landmarks_list=landmarks_list,
                faces_data=faces_data,
                alert_track_ids=alert_track_ids,
            )
            if annotated_frame is not None:
                cv2.putText(annotated_frame, hud_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow("Cam-Security | Visão Real", annotated_frame)

            if cv2.waitKey(1) == 27:
                sys_logger.info("Encerrando a pedido do usuário.")
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
