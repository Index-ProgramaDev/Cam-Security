
import cv2
import numpy as np
from utils.math_utils import calcular_distancia


def desenhar_anotacoes_mediapipe(frame, detection_result):
    if frame is None:
        return None

    canvas = frame.copy()

    try:
        from mediapipe.tasks.python import vision
        mp_drawing = vision.drawing_utils
        mp_drawing_styles = vision.drawing_styles

        for person in detection_result.get("people", []):
            if person.get("landmarks"):
                mp_drawing.draw_landmarks(
                    canvas,
                    person["landmarks"],
                    vision.PoseLandmarksConnections.POSE_LANDMARKS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
    except Exception as e:
        import traceback
        traceback.print_exc()
        pass

    alert_triggered = detection_result.get("alert_triggered", False)
    people = detection_result.get("people", [])

    # Draw bounding boxes and IDs
    for person in people:
        box = person.get("box")
        if box:
            x1, y1, x2, y2 = map(int, box)
            color = (0, 0, 255) if alert_triggered else (0, 255, 0)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f"Pessoa #{person.get('track_id')}"
            cv2.rectangle(canvas, (x1, y1 - 30), (x1 + 120, y1), color, -1)
            cv2.putText(canvas, label, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Draw lines and distances between pairs of people
    for i in range(len(people)):
        p1 = people[i]
        for j in range(i+1, len(people)):
            p2 = people[j]
            center1 = p1.get("center")
            center2 = p2.get("center")
            if center1 and center2:
                dist = calcular_distancia(center1, center2)
                pt1 = (int(center1[0]), int(center1[1]))
                pt2 = (int(center2[0]), int(center2[1]))
                
                # Line color changes based on distance (red if close, green if far)
                line_color = (0, 255, 0) if dist > 150 else (0, 0, 255)
                cv2.line(canvas, pt1, pt2, line_color, 2)
                
                # Draw distance text in the middle of the line
                mid_x = int((center1[0] + center2[0]) / 2)
                mid_y = int((center1[1] + center2[1]) / 2)
                dist_label = f"{dist:.1f} px"
                cv2.rectangle(canvas, (mid_x - 45, mid_y - 15), (mid_x + 45, mid_y + 10), (255, 255, 255), -1)
                cv2.putText(canvas, dist_label, (mid_x - 40, mid_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    return canvas

