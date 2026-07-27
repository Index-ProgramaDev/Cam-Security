import cv2
import time
import sys
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
    last_frame_time = time.time()

    def try_reid(person_crop):
        insights = face_capture.capture_face_insights(person_crop)
        if insights and insights["embedding"] is not None:
            return face_reid.match_embedding(insights["embedding"])
        return None

    try:
        while camera.running:
            now = time.time()
            delta_t = now - last_frame_time
            last_frame_time = now

            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]

            # a) Detecção de Poses (Esqueleto)
            pose_result = pose_detector.process(frame)
            landmarks_list = pose_result.get("pose_landmarks", [])

            # b) Detecção de Pessoas (com NMS ativado)
            detected_persons = person_detector.detect_persons(frame, pose_landmarks_list=landmarks_list)

            # c) Rastreamento (Atribuição Gulosa por IoU + Centro + 5 min TTL)
            tracks = object_tracker.update(detected_persons, face_reid_callback=try_reid, frame=frame)

            # d) Detecção Facial
            faces_data = face_detector.detect_faces(frame)

            # e) Análise de Distância e Colisão entre Pares de Pessoas com Histórico Mínimo
            track_ids = list(tracks.keys())
            has_multiple_people = len(track_ids) >= 2
            collision_detected = False
            min_dist_label = "LONGE"

            if has_multiple_people:
                for i in range(len(track_ids)):
                    for j in range(i + 1, len(track_ids)):
                        idA, idB = track_ids[i], track_ids[j]
                        infoA, infoB = tracks[idA], tracks[idB]

                        # Exige que AMBOS os tracks tenham idade (age) >= 10 frames para eventos interativos
                        both_established = (infoA.get("age", 0) >= MIN_TRACK_AGE_FOR_EVENTS) and (infoB.get("age", 0) >= MIN_TRACK_AGE_FOR_EVENTS)
                        if not both_established:
                            continue

                        boxA, boxB = infoA["box"], infoB["box"]
                        rel_dist, dist_label = calculate_relative_distance(boxA, boxB)
                        is_colliding = detect_collision(boxA, boxB)

                        if is_colliding:
                            collision_detected = True
                            risk = calculate_risk_score("COLISAO", dist_label, 0.0, True)
                            alert_manager.trigger_alert("COLISAO", idA, risk_score=risk, description=f"Colisão entre #{idA} e #{idB}")

                        if dist_label == "PERTO":
                            min_dist_label = "PERTO"
                        elif dist_label == "MEDIO" and min_dist_label != "PERTO":
                            min_dist_label = "MEDIO"

            # f) Processamento por Track Ativo
            alert_track_ids = []

            for idx, (track_id, info) in enumerate(tracks.items()):
                box = info["box"]
                track_age = info.get("age", 0)

                curr_centroid = calculate_center(box)
                prev_centroid = prev_centroids.get(track_id)
                body_vel = calculate_velocity(curr_centroid, prev_centroid, delta_t)
                prev_centroids[track_id] = curr_centroid

                # Atualiza embedding facial
                x1, y1, x2, y2 = map(int, box)
                crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                if crop.size > 0:
                    insights = face_capture.capture_face_insights(crop)
                    if insights and insights["embedding"] is not None:
                        face_storage.save_embedding(track_id, insights["embedding"])

                # Avaliação de Poses e Soco
                if landmarks_list and idx < len(landmarks_list):
                    landmarks = landmarks_list[idx]
                    r_shoulder, r_elbow, r_wrist = landmarks[12], landmarks[14], landmarks[16]
                    l_shoulder, l_elbow, l_wrist = landmarks[11], landmarks[13], landmarks[15]

                    # Velocidade do pulso
                    curr_wrist_pos = (r_wrist.x * w, r_wrist.y * h)
                    prev_wrist_pos = prev_wrists.get(track_id)
                    wrist_vel = calculate_velocity(curr_wrist_pos, prev_wrist_pos, delta_t)
                    prev_wrists[track_id] = curr_wrist_pos

                    # Ângulo do cotovelo
                    r_angle = calculate_arm_angle(r_shoulder, r_elbow, r_wrist)
                    l_angle = calculate_arm_angle(l_shoulder, l_elbow, l_wrist)
                    max_angle = max(r_angle, l_angle)

                    # SOCO exige: 2+ pessoas, proximidade/colisão E histórico do track >= 10 frames
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
                        # Poses proibidas gerais (ARM_RAISED, FALLEN) com buffer de 5 frames
                        is_forbidden, pose_name = pose_estimator.evaluate(landmarks, track_id=track_id)
                        if is_forbidden:
                            alert_track_ids.append(track_id)
                            object_tracker.set_trigger(track_id)
                            risk = calculate_risk_score(pose_name, min_dist_label, body_vel, collision_detected)
                            alert_manager.trigger_alert(pose_name, track_id, risk_score=risk, description=f"Pose proibida: {pose_name}")

            # g) Renderização Visual com HUD
            annotated_frame = person_detector.draw_annotations(
                frame,
                tracks=tracks,
                pose_landmarks_list=landmarks_list,
                faces_data=faces_data,
                alert_track_ids=alert_track_ids
            )

            if annotated_frame is not None:
                hud_text = f"Pessoas: {len(tracks)} | Dist: {min_dist_label} | Colisao: {'SIM' if collision_detected else 'NAO'}"
                cv2.putText(annotated_frame, hud_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow("Cam-Security | Visão Real", annotated_frame)

            if cv2.waitKey(1) == 27:  # ESC
                sys_logger.info("Encerrando a pedido do usuário.")
                break

    except KeyboardInterrupt:
        sys_logger.info("Interrompido via KeyboardInterrupt.")
    except Exception as e:
        sys_logger.error(f"Erro no loop principal: {e}")
        import traceback
        traceback.print_exc()

    finally:
        pose_detector.close()
        face_detector.close()
        face_capture.close()
        camera.stop()
        cv2.destroyAllWindows()
        sys_logger.info("=== Cam-Security Finalizado ===")

if __name__ == "__main__":
    main()
