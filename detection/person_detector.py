import cv2
import numpy as np
from utils.logger import sys_logger
from tracking.object_tracker import compute_iou

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28)
]

def apply_nms(boxes, iou_threshold=0.45):
    """
    Aplica Non-Maximum Suppression (NMS) para eliminar caixas duplicadas da mesma pessoa no mesmo frame.
    """
    if not boxes:
        return []

    sorted_boxes = sorted(boxes, key=lambda b: b.get("confidence", 0.0), reverse=True)
    selected_boxes = []

    for item in sorted_boxes:
        boxA = item["box"]
        keep = True
        for sel in selected_boxes:
            boxB = sel["box"]
            iou = compute_iou(boxA, boxB)
            if iou > iou_threshold:
                keep = False
                break
        if keep:
            selected_boxes.append(item)

    return selected_boxes

class PersonDetector:
    def __init__(self):
        self.yolo_model = None
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO("yolov8n.pt")
            sys_logger.info("YOLOv8n inicializado com sucesso.")
        except Exception as e:
            sys_logger.warning(f"Erro ao carregar YOLOv8n ({e}). Usando fallback de pose.")

    def detect_persons(self, frame, pose_landmarks_list=None):
        if frame is None:
            return []

        raw_boxes = []
        h, w = frame.shape[:2]

        if self.yolo_model:
            try:
                results = self.yolo_model(frame, verbose=False, conf=0.55)[0]
                for det in results.boxes:
                    cls_id = int(det.cls[0])
                    if cls_id == 0:  # Classe 0 = Pessoa em COCO
                        x1, y1, x2, y2 = det.xyxy[0].cpu().numpy().astype(int)
                        conf = float(det.conf[0])
                        raw_boxes.append({"box": [int(x1), int(y1), int(x2), int(y2)], "confidence": conf})
            except Exception as e:
                sys_logger.error(f"Erro na detecção YOLO: {e}")

        # Se YOLO não detectou, usa fallback de pose
        if not raw_boxes and pose_landmarks_list:
            for landmarks in pose_landmarks_list:
                xs = [lm.x * w for lm in landmarks]
                ys = [lm.y * h for lm in landmarks]
                x1, y1 = max(0, int(min(xs)) - 20), max(0, int(min(ys)) - 20)
                x2, y2 = min(w, int(max(xs)) + 20), min(h, int(max(ys)) + 20)
                raw_boxes.append({"box": [x1, y1, x2, y2], "confidence": 0.9})

        # Aplica NMS estrito para garantir que 1 pessoa NUNCA vire 2 caixas no mesmo frame
        filtered_boxes = apply_nms(raw_boxes, iou_threshold=0.45)
        return filtered_boxes

    def draw_annotations(self, frame, tracks, pose_landmarks_list=None, faces_data=None, alert_track_ids=None):
        if frame is None:
            return None

        canvas = frame.copy()
        h, w, _ = frame.shape
        alert_track_ids = alert_track_ids or []

        # 1. Esqueleto corporal de Pose
        if pose_landmarks_list:
            for landmarks in pose_landmarks_list:
                for (start_idx, end_idx) in POSE_CONNECTIONS:
                    if start_idx < len(landmarks) and end_idx < len(landmarks):
                        x1 = int(landmarks[start_idx].x * w)
                        y1 = int(landmarks[start_idx].y * h)
                        x2 = int(landmarks[end_idx].x * w)
                        y2 = int(landmarks[end_idx].y * h)
                        cv2.line(canvas, (x1, y1), (x2, y2), (255, 255, 0), 2)
                for lm in landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(canvas, (cx, cy), 3, (0, 255, 255), -1)

        # 2. Face Mesh e caixa do rosto
        if faces_data:
            for face in faces_data:
                box = face.get("box")
                if box:
                    fx, fy, fw, fh = box
                    cv2.rectangle(canvas, (fx, fy), (fx + fw, fy + fh), (255, 255, 0), 1)
                lms = face.get("landmarks")
                if lms:
                    for lm in lms:
                        lx, ly = int(lm.x * w), int(lm.y * h)
                        cv2.circle(canvas, (lx, ly), 1, (0, 255, 0), -1)

        # 3. Bounding Box da pessoa e ID/Alerta
        for track_id, info in tracks.items():
            box = info.get("box")
            if not box or len(box) < 4:
                continue

            x1, y1, x2, y2 = map(int, box)
            is_alert = track_id in alert_track_ids
            color = (0, 0, 255) if is_alert else (0, 255, 0)

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f"Pessoa #{track_id}" + (" [ALERTA PROIBIDO!]" if is_alert else "")
            cv2.rectangle(canvas, (x1, y1 - 25), (x1 + len(label) * 10, y1), color, -1)
            cv2.putText(canvas, label, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return canvas
