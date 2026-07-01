import cv2
import numpy as np
from utils.math_utils import calcular_distancia, calcular_centro


POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]


def desenhar_anotacoes_mediapipe(frame, detection_result):
    if frame is None:
        return None

    canvas = frame.copy()
    h, w, _ = frame.shape

    alert_triggered = detection_result.get("alert_triggered", False)
    people = detection_result.get("people", [])

    for person in people:
        landmarks = person.get("landmarks")
        if landmarks:
            for (start_idx, end_idx) in POSE_CONNECTIONS:
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    x1 = int(landmarks[start_idx].x * w)
                    y1 = int(landmarks[start_idx].y * h)
                    x2 = int(landmarks[end_idx].x * w)
                    y2 = int(landmarks[end_idx].y * h)
                    cv2.line(canvas, (x1, y1), (x2, y2), (255, 0, 0), 2)

        box = person.get("box")
        if box:
            x1, y1, x2, y2 = map(int, box)
            color = (0, 0, 255) if alert_triggered else (0, 255, 0)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f"Pessoa #{person.get('track_id')}"
            cv2.rectangle(canvas, (x1, y1 - 30), (x1 + 120, y1), color, -1)
            cv2.putText(canvas, label, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return canvas


def desenhar_status_deteccao(canvas, detection_result, hand_open_frames=0, frames_required=30):
    return canvas
