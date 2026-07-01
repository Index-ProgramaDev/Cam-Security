
import cv2
import numpy as np
import mediapipe as mp
from utils.math_utils import calcular_distancia, calcular_centro


def desenhar_anotacoes_mediapipe(frame, detection_result):
    if frame is None:
        return None

    canvas = frame.copy()

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    alert_triggered = detection_result.get("alert_triggered", False)
    people = detection_result.get("people", [])

    # Draw skeletons, bounding boxes and IDs
    for person in people:
        if person.get("landmarks"):
            mp_drawing.draw_landmarks(
                canvas,
                person["landmarks"],
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2)
            )
        
        box = person.get("box")
        if box:
            x1, y1, x2, y2 = map(int, box)
            color = (0, 0, 255) if alert_triggered else (0, 255, 0)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f"Pessoa #{person.get('track_id')}"
            cv2.rectangle(canvas, (x1, y1 - 30), (x1 + 120, y1), color, -1)
            cv2.putText(canvas, label, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return canvas

