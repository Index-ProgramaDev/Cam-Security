import cv2
from utils.logger import sys_logger
from tracking.object_tracker import compute_iou

CROP_PAD_RATIO       = 0.18
CROP_EXTRA_TOP       = 0.20
MIN_LANDMARK_VISIBILITY = 0.30

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
    if not boxes:
        return []
    sorted_boxes = sorted(boxes, key=lambda b: b.get("confidence", 0.0), reverse=True)
    selected = []
    for item in sorted_boxes:
        if all(compute_iou(item["box"], s["box"]) <= iou_threshold for s in selected):
            selected.append(item)
    return selected


def expand_person_box(box, frame_w, frame_h, pad_ratio=CROP_PAD_RATIO, extra_top=CROP_EXTRA_TOP):
    x1, y1, x2, y2 = [int(v) for v in box[:4]]
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    pad_x, pad_y = int(bw * pad_ratio), int(bh * pad_ratio)
    top = int(bh * extra_top)
    return [
        max(0, x1 - pad_x),
        max(0, y1 - pad_y - top),
        min(frame_w, x2 + pad_x),
        min(frame_h, y2 + pad_y),
    ]


def crop_person(frame, box, pad=True):
    h, w = frame.shape[:2]
    crop_box = expand_person_box(box, w, h) if pad else [int(v) for v in box[:4]]
    x1, y1, x2, y2 = crop_box
    return frame[y1:y2, x1:x2], crop_box


class PersonDetector:
    def __init__(self):
        self.yolo_model = None
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO("yolov8n.pt")
            sys_logger.info("YOLOv8n inicializado com sucesso.")
        except Exception as e:
            sys_logger.warning(f"Erro ao carregar YOLOv8n ({e}). Sem detecção de pessoas.")

    def detect_persons(self, frame, pose_landmarks_list=None):
        if frame is None:
            return []

        raw_boxes = []
        h, w = frame.shape[:2]

        if self.yolo_model:
            try:
                results = self.yolo_model(frame, imgsz=512, verbose=False, conf=0.50)[0]
                for det in results.boxes:
                    if int(det.cls[0]) == 0:
                        x1, y1, x2, y2 = det.xyxy[0].cpu().numpy().astype(int)
                        raw_boxes.append({
                            "box":        [int(x1), int(y1), int(x2), int(y2)],
                            "confidence": float(det.conf[0]),
                        })
            except Exception as e:
                sys_logger.error(f"Erro na detecção YOLO: {e}")

        # Fallback por landmarks de pose quando YOLO não detecta nada
        if not raw_boxes and pose_landmarks_list:
            for landmarks in pose_landmarks_list:
                xs = [lm.x * w for lm in landmarks]
                ys = [lm.y * h for lm in landmarks]
                x1 = max(0, int(min(xs)) - 20)
                y1 = max(0, int(min(ys)) - 20)
                x2 = min(w, int(max(xs)) + 20)
                y2 = min(h, int(max(ys)) + 20)
                raw_boxes.append({"box": [x1, y1, x2, y2], "confidence": 0.9})

        return apply_nms(raw_boxes, iou_threshold=0.45)

    def draw_annotations(self, frame, tracks, pose_landmarks_list=None,
                         faces_data=None, alert_track_ids=None):
        if frame is None:
            return None

        canvas = frame.copy()
        h, w   = frame.shape[:2]
        alert_track_ids = alert_track_ids or []

        for track_id, info in tracks.items():
            box = info.get("box")
            if not box or len(box) < 4:
                continue

            x1, y1, x2, y2 = map(int, box)
            is_alert = track_id in alert_track_ids
            color    = (0, 0, 255) if is_alert else (0, 255, 0)

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            identity    = info.get("identity")
            face_status = info.get("face_status") or ""
            name = f"Pessoa #{track_id} [{identity}]" if identity is not None else f"Pessoa #{track_id}"
            if is_alert:
                name += " [ALERTA]"
            if face_status and face_status != "novo":
                name += f" | {face_status}"

            label_w = max(80, min(w - x1, 10 + len(name) * 8))
            cv2.rectangle(canvas, (x1, max(0, y1 - 22)), (x1 + label_w, y1), color, -1)
            cv2.putText(canvas, name, (x1 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            pose = info.get("pose")
            if pose:
                self._draw_skeleton(canvas, pose, w, h)

            face_box  = info.get("face_box")
            face_conf = info.get("face_confidence")
            if face_box and len(face_box) >= 4:
                fx, fy, fw, fh = [int(v) for v in face_box[:4]]
                cv2.rectangle(canvas, (fx, fy), (fx + fw, fy + fh), (255, 200, 0), 2)
                label = f"face {face_conf:.2f}" if face_conf is not None else "face"
                cv2.putText(canvas, label, (fx, max(12, fy - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

        # Compatibilidade: lista solta de poses (não usada no pipeline atual)
        if pose_landmarks_list:
            for landmarks in pose_landmarks_list:
                self._draw_skeleton(canvas, landmarks, w, h)

        return canvas

    def _draw_skeleton(self, canvas, landmarks, w, h):
        def _vis(lm):
            return getattr(lm, "visibility", 1.0)

        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx >= len(landmarks) or end_idx >= len(landmarks):
                continue
            a, b = landmarks[start_idx], landmarks[end_idx]
            if _vis(a) < MIN_LANDMARK_VISIBILITY or _vis(b) < MIN_LANDMARK_VISIBILITY:
                continue
            p1 = (int(a.x * w), int(a.y * h))
            p2 = (int(b.x * w), int(b.y * h))
            if not (0 <= p1[0] < w and 0 <= p1[1] < h and 0 <= p2[0] < w and 0 <= p2[1] < h):
                continue
            cv2.line(canvas, p1, p2, (255, 255, 0), 2)

        for lm in landmarks:
            if _vis(lm) < MIN_LANDMARK_VISIBILITY:
                continue
            px, py = int(lm.x * w), int(lm.y * h)
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(canvas, (px, py), 3, (0, 255, 255), -1)
