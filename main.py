
import time
import cv2
from loguru import logger

from camera.capture import CameraCapture
from detection.mediapipe_detector import MediaPipeDetector
from events.event_logger import EventLogger
from events.alerts import AlertManager
from ui.draw_boxes import desenhar_anotacoes_mediapipe


def main():
    logger.info("=== Iniciando Sistema Cam-Security Simplificado com MediaPipe ===")

    cap_device = CameraCapture()
    cap_device.start()

    detector = MediaPipeDetector()

    event_logger = EventLogger()
    alert_manager = AlertManager(event_logger, cooldown_seconds=3.0)

    is_synth = cap_device.is_synthetic_active
    logger.info(f"Câmera iniciada no modo: {'SIMULADO (Sintético)' if is_synth else 'REAL (Física)'}")
    logger.info("Pressione a tecla 'ESC' na janela de vídeo para fechar o programa.")
    logger.info("Abra a mão completamente para disparar um alerta sonoro!")

    alert_sound_played = False

    while cap_device.running:
        try:
            frame = cap_device.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            detection_result = detector.process(frame)

            current_alert_triggered = detection_result.get("alert_triggered", False)
            if current_alert_triggered and not alert_sound_played:
                track_id = detection_result.get("track_id", 0)
                logger.warning("ALERTA PERMANENTE ATIVADO!")
                
                alert_manager.trigger_alert(
                    event_type="MAO_ABERTA",
                    track_id=track_id,
                    risk_score=100,
                    description="Mão completamente aberta detectada (alerta permanente ativado)"
                )

                alert_manager.play_alert_sound()
                time.sleep(0.3)
                alert_manager.play_alert_sound()
                
                alert_sound_played = True

            annotated_frame = desenhar_anotacoes_mediapipe(frame, detection_result)

            if annotated_frame is not None:
                cv2.imshow("Cam-Security | MediaPipe", annotated_frame)

            if cv2.waitKey(1) & 0xFF == 27:
                logger.info("Tecla ESC pressionada. Encerrando monitoramento...")
                break

        except Exception as loop_e:
            logger.error(f"Erro inesperado no loop principal: {loop_e}")
            import traceback
            traceback.print_exc()
            time.sleep(1.0)

    detector.close()
    cap_device.stop()
    cv2.destroyAllWindows()
    logger.info("=== Cam-Security Finalizado ===")


if __name__ == "__main__":
    main()

