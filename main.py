import time
import cv2
from loguru import logger

from camera.capture import CameraCapture
from detection.mediapipe_detector import MediaPipeDetector
from events.event_logger import EventLogger
from events.alerts import AlertManager
from ui.draw_boxes import desenhar_anotacoes_mediapipe, desenhar_status_deteccao


def main():
    logger.info("=== Iniciando Sistema Cam-Security ===")

    cap_device = CameraCapture()
    cap_device.start()

    detector = MediaPipeDetector()

    event_logger = EventLogger()
    alert_manager = AlertManager(event_logger, cooldown_seconds=3.0)

    is_synth = cap_device.is_synthetic_active
    logger.info(f"Câmera: {'SINTÉTICA' if is_synth else 'FÍSICA'}")
    logger.info("ESC para encerrar | ENTER para resetar alerta")

    alert_sound_played = False

    try:
        while cap_device.running:
            try:
                frame = cap_device.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                detection_result = detector.process(frame)

                current_alert_triggered = detection_result.get("alert_triggered", False)
                if current_alert_triggered and not alert_sound_played:
                    people = detection_result.get("people", [])
                    track_id = people[0]["track_id"] if people else 0
                    logger.warning("ALERTA ATIVADO!")

                    alert_manager.trigger_alert(
                        event_type="BRACO_LEVANTADO",
                        track_id=track_id,
                        risk_score=100,
                        description="Braço levantado alto detectado"
                    )

                    alert_manager.play_alert_sound()
                    time.sleep(0.3)
                    alert_manager.play_alert_sound()

                    alert_sound_played = True

                annotated_frame = desenhar_anotacoes_mediapipe(frame, detection_result)
                if annotated_frame is not None:
                    annotated_frame = desenhar_status_deteccao(
                        annotated_frame,
                        detection_result,
                        hand_open_frames=detector.hand_open_frames,
                        frames_required=detector.frames_required_open
                    )

                if annotated_frame is not None:
                    cv2.imshow("Cam-Security | MediaPipe", annotated_frame)

                key = cv2.waitKey(1)
                if key == 27:
                    logger.info("Encerrando...")
                    break
                elif key == 13:
                    if detector.alert_triggered:
                        detector.reset_alert()
                        alert_sound_played = False

            except KeyboardInterrupt:
                raise
            except Exception as loop_e:
                logger.error(f"Erro no loop: {loop_e}")
                import traceback
                traceback.print_exc()
                time.sleep(1.0)

    except KeyboardInterrupt:
        logger.info("Interrompido pelo usuário.")

    finally:
        detector.close()
        cap_device.stop()
        cv2.destroyAllWindows()
        logger.info("=== Cam-Security Finalizado ===")


if __name__ == "__main__":
    main()
