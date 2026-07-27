import cv2
import numpy as np
from utils.logger import sys_logger

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28)
]

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

        boxes = []
        h, w = frame.shape[:2]

        if self.yolo_model:
            try:
                # Confiança aumentada para 0.55 para filtrar falsos positivos
                results = self.yolo_model(frame, verbose=False, conf=0.55)[0]
                for det in results.boxes:
                    cls_id = int(det.cls[0])
                    if cls_id == 0:  # Classe 0 = Pessoa
                        x1, y1, x2, y2 = det.xyxy[0].cpu().numpy().astype(int)
                        conf = float(det.conf[0])
                        boxes.append({"box": [int(x1), int(y1), int(x2), int(y2)], "confidence": conf})
                if boxes:
                    return boxes
            except Exception as e:
                sys_logger.error(f"Erro na detecção YOLO: {e}")

        if pose_landmarks_list:
            for landmarks in pose_landmarks_list:
                xs = [lm.x * w for lm in landmarks]
                ys = [lm.y * h for lm in landmarks]
                x1, y1 = max(0, int(min(xs)) - 20), max(0, int(min(ys)) - 20)
                x2, y2 = min(w, int(max(xs)) + 20), min(h, int(max(ys)) + 20)
                boxes.append({"box": [x1, y1, x2, y2], "confidence": 0.9})

        return boxes

    def draw_annotations(self, frame, tracks, pose_landmarks_list=None, faces_data=None, alert_track_ids=None):
        """
        Desenha no frame:
        1. Bounding Box da pessoa e rótulo de ID/Alerta.
        2. Esqueleto corporal de Pose (linhas azuis/ciano).
        3. Face Mesh e caixa do rosto (pontos amarelos).
        """
        if frame is None:
            return None

        canvas = frame.copy()
        h, w, _ = frame.shape
        alert_track_ids = alert_track_ids or []

        # 1. Desenha o esqueleto corporal de Pose
        if pose_landmarks_list:
            for landmarks in pose_landmarks_list:
                # Desenha conexões dos membros
                for (start_idx, end_idx) in POSE_CONNECTIONS:
                    if start_idx < len(landmarks) and end_idx < len(landmarks):
                        x1 = int(landmarks[start_idx].x * w)
                        y1 = int(landmarks[start_idx].y * h)
                        x2 = int(landmarks[end_idx].x * w)
                        y2 = int(landmarks[end_idx].y * h)
                        cv2.line(canvas, (x1, y1), (x2, y2), (255, 255, 0), 2)
                # Desenha articulações
                for lm in landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(canvas, (cx, cy), 3, (0, 255, 255), -1)

        # 2. Desenha o Face Mesh e Bounding Box do Rosto
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

        # 3. Desenha a caixa delimitadora da pessoa e o ID/Alerta
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
