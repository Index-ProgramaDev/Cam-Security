import cv2
from utils.logger import sys_logger
from tracking.object_tracker import compute_iou

# Padding relativo à largura/altura da bbox para incluir cabeça e pés inteiros.
# 0.18 lateral + 0.20 extra no topo cobre a maioria das poses sem exagero.
CROP_PAD_RATIO = 0.18
CROP_EXTRA_TOP = 0.20
MIN_LANDMARK_VISIBILITY = 0.30

# Topologia completa dos 33 landmarks do MediaPipe Pose.
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
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


def expand_person_box(box, frame_w, frame_h, pad_ratio=CROP_PAD_RATIO, extra_top=CROP_EXTRA_TOP):
    """Expande a bbox YOLO para o crop incluir cabeça e pés sem exagero."""
    x1, y1, x2, y2 = [int(v) for v in box[:4]]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)
    top = int(bh * extra_top)
    nx1 = max(0, x1 - pad_x)
    ny1 = max(0, y1 - pad_y - top)
    nx2 = min(frame_w, x2 + pad_x)
    ny2 = min(frame_h, y2 + pad_y)
    return [nx1, ny1, nx2, ny2]


def crop_person(frame, box, pad=True):
    """Retorna (crop, crop_box) a partir da bbox da pessoa."""
    h, w = frame.shape[:2]
    crop_box = expand_person_box(box, w, h) if pad else [int(v) for v in box[:4]]
    x1, y1, x2, y2 = crop_box
    crop = frame[y1:y2, x1:x2]
    return crop, crop_box


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
                results = self.yolo_model(frame, verbose=False, conf=0.50)[0]
                for det in results.boxes:
                    cls_id = int(det.cls[0])
                    if cls_id == 0:
                        x1, y1, x2, y2 = det.xyxy[0].cpu().numpy().astype(int)
                        conf = float(det.conf[0])
                        raw_boxes.append({"box": [int(x1), int(y1), int(x2), int(y2)], "confidence": conf})
            except Exception as e:
                sys_logger.error(f"Erro na detecção YOLO: {e}")

        if not raw_boxes and pose_landmarks_list:
            for landmarks in pose_landmarks_list:
                xs = [lm.x * w for lm in landmarks]
                ys = [lm.y * h for lm in landmarks]
                x1, y1 = max(0, int(min(xs)) - 20), max(0, int(min(ys)) - 20)
                x2, y2 = min(w, int(max(xs)) + 20), min(h, int(max(ys)) + 20)
                raw_boxes.append({"box": [x1, y1, x2, y2], "confidence": 0.9})

        filtered_boxes = apply_nms(raw_boxes, iou_threshold=0.45)
        return filtered_boxes

    def draw_annotations(self, frame, tracks, pose_landmarks_list=None, faces_data=None, alert_track_ids=None):
        """
        Renderiza anotações sobre o frame.
        A pose de cada track é lida de info['pose'] (já mapeada para o frame).
        pose_landmarks_list e faces_data mantidos apenas para compatibilidade retroativa.
        """
        if frame is None:
            return None

        canvas = frame.copy()
        h, w, _ = frame.shape
        alert_track_ids = alert_track_ids or []

        for track_id, info in tracks.items():
            box = info.get("box")
            if not box or len(box) < 4:
                continue

            x1, y1, x2, y2 = map(int, box)
            is_alert = track_id in alert_track_ids
            color = (0, 0, 255) if is_alert else (0, 255, 0)

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            identity = info.get("identity")
            face_status = info.get("face_status") or ""
            if identity is not None:
                name = f"Pessoa #{track_id} [{identity}]"
            else:
                name = f"Pessoa #{track_id}"
            if is_alert:
                name += " [ALERTA]"
            if face_status and face_status != "novo":
                name += f" | {face_status}"

            label_w = max(80, min(w - x1, 10 + len(name) * 8))
            cv2.rectangle(canvas, (x1, max(0, y1 - 22)), (x1 + label_w, y1), color, -1)
            cv2.putText(canvas, name, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            # Skeleton — usa a pose já mapeada para o frame (armazenada no track)
            pose = info.get("pose")
            if pose:
                self._draw_skeleton(canvas, pose, w, h)

            # Face bounding box (coordenadas absolutas no frame)
            face_box = info.get("face_box")
            face_conf = info.get("face_confidence")
            if face_box and len(face_box) >= 4:
                fx, fy, fw, fh = [int(v) for v in face_box[:4]]
                cv2.rectangle(canvas, (fx, fy), (fx + fw, fy + fh), (255, 200, 0), 2)
                conf_txt = f"face {face_conf:.2f}" if face_conf is not None else "face"
                cv2.putText(canvas, conf_txt, (fx, max(12, fy - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

        # Compatibilidade retroativa: lista solta de poses/faces (não usado no pipeline v3)
        if pose_landmarks_list:
            for landmarks in pose_landmarks_list:
                self._draw_skeleton(canvas, landmarks, w, h)

        return canvas

    def _draw_skeleton(self, canvas, landmarks, w, h):
        """
        Desenha o skeleton no canvas.
        Espera landmarks com coordenadas normalizadas no espaço do frame (0..1),
        já convertidas por map_crop_landmarks_to_frame().
        """
        def _vis(lm):
            return getattr(lm, "visibility", 1.0)

        for (start_idx, end_idx) in POSE_CONNECTIONS:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                a, b = landmarks[start_idx], landmarks[end_idx]
                if _vis(a) < MIN_LANDMARK_VISIBILITY or _vis(b) < MIN_LANDMARK_VISIBILITY:
                    continue
                # x/y já estão normalizados para o frame completo
                p1 = (int(a.x * w), int(a.y * h))
                p2 = (int(b.x * w), int(b.y * h))
                # Descarta pontos fora dos limites do canvas (pode ocorrer com padding agressivo)
                if not (0 <= p1[0] < w and 0 <= p1[1] < h and 0 <= p2[0] < w and 0 <= p2[1] < h):
                    continue
                cv2.line(canvas, p1, p2, (255, 255, 0), 2)
        for lm in landmarks:
            if _vis(lm) < MIN_LANDMARK_VISIBILITY:
                continue
            px, py = int(lm.x * w), int(lm.y * h)
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(canvas, (px, py), 3, (0, 255, 255), -1)
