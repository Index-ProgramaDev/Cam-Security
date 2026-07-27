import cv2
import time
import sys
from utils.logger import sys_logger
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

def main():
    sys_logger.info("=== Iniciando Backend Cam-Security ===")

    # 1. Câmera Real
    camera = CameraCapture()
    try:
        camera.start()
    except Exception as e:
        sys_logger.error(f"Erro na inicialização da câmera: {e}")
        return

    # 2. Rastreamento e IDManager
    id_manager = IDManager()
    object_tracker = ObjectTracker(id_manager=id_manager, ttl_seconds=300.0)

    # 3. Biometria Facial e Face Mesh
    face_detector = FaceDetector()
    face_storage = FaceStorage()
    face_capture = FaceCapture()
    face_reid = FaceReID(face_storage=face_storage)

    # 4. Detecção de Pessoas e Pose
    person_detector = PersonDetector()
    pose_detector = MediaPipeDetector()
    pose_estimator = PoseEstimator()

    # 5. Eventos e Alertas
    event_logger = EventLogger()
    notification_dispatcher = NotificationDispatcher()
    alert_manager = AlertManager(event_logger=event_logger, notification_dispatcher=notification_dispatcher)

    sys_logger.info("Sistema ativo com PoseLandmarker e Face Mesh. Pressione ESC para encerrar.")

    def try_reid(person_crop):
        insights = face_capture.capture_face_insights(person_crop)
        if insights and insights["embedding"] is not None:
            matched_id = face_reid.match_embedding(insights["embedding"])
            return matched_id
        return None

    try:
        while camera.running:
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # a) Detecção de Poses (Esqueleto)
            pose_result = pose_detector.process(frame)
            landmarks_list = pose_result.get("pose_landmarks", [])

            # b) Detecção de Pessoas (YOLOv8 + Fallback)
            detected_persons = person_detector.detect_persons(frame, pose_landmarks_list=landmarks_list)

            # c) Rastreamento (IoU + FaceReID + 5 min TTL)
            tracks = object_tracker.update(detected_persons, face_reid_callback=try_reid, frame=frame)

            # d) Detecção Facial e Mesh
            faces_data = face_detector.detect_faces(frame)

            # e) Atualização de embeddings faciais no FaceStorage para cada track ativo
            for track_id, info in tracks.items():
                box = info["box"]
                x1, y1, x2, y2 = map(int, box)
                crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                if crop.size > 0:
                    insights = face_capture.capture_face_insights(crop)
                    if insights and insights["embedding"] is not None:
                        face_storage.save_embedding(track_id, insights["embedding"])

            # f) Avaliação de Poses Proibidas e Disparo de Alertas
            alert_track_ids = []
            if landmarks_list:
                for idx, landmarks in enumerate(landmarks_list):
                    is_forbidden, pose_name = pose_estimator.evaluate(landmarks)
                    if is_forbidden:
                        active_ids = list(tracks.keys())
                        target_id = active_ids[idx] if idx < len(active_ids) else 1
                        alert_track_ids.append(target_id)
                        object_tracker.set_trigger(target_id)
                        
                        alert_manager.trigger_alert(
                            event_type=pose_name,
                            track_id=target_id,
                            risk_score=100,
                            description=f"Pose proibida detectada: {pose_name}"
                        )

            # g) Renderização Visual (Pessoas, Esqueleto, Face Mesh e Alertas)
            annotated_frame = person_detector.draw_annotations(
                frame,
                tracks=tracks,
                pose_landmarks_list=landmarks_list,
                faces_data=faces_data,
                alert_track_ids=alert_track_ids
            )

            if annotated_frame is not None:
                cv2.imshow("Cam-Security | Visão Real", annotated_frame)

            if cv2.waitKey(1) == 27:  # ESC
                sys_logger.info("Encerrando a pedido do usuário.")
                break

    except KeyboardInterrupt:
        sys_logger.info("Interrompido pelo usuário.")
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
