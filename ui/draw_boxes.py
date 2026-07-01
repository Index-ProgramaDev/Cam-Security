
import cv2
import numpy as np


def desenhar_anotacoes_mediapipe(frame, detection_result):
    if frame is None:
        return None

    canvas = frame.copy()

    try:
        from mediapipe.tasks.python import vision
        mp_drawing = vision.drawing_utils
        mp_drawing_styles = vision.drawing_styles

        if detection_result.get("pose_landmarks"):
            mp_drawing.draw_landmarks(
                canvas,
                detection_result["pose_landmarks"],
                vision.PoseLandmarksConnections.POSE_LANDMARKS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

    except Exception as e:
        import traceback
        traceback.print_exc()
        pass

    alert_triggered = detection_result.get("alert_triggered", False)
    if detection_result.get("box"):
        x1, y1, x2, y2 = detection_result["box"]
        
        color = (0, 0, 255) if alert_triggered else (0, 255, 0)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        track_id = detection_result.get("track_id")
        if alert_triggered:
            label = f"Pessoa #{track_id} - ALERTA ATIVADO!"
        else:
            label = f"Pessoa #{track_id}"

        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(canvas, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), color, -1)
        cv2.putText(canvas, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return canvas

