import time
import math
from utils.logger import sys_logger

# TTL de propagação do face_box sem nova detecção (suaviza flickering)
FACE_BOX_TTL = 0.25


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter)


class ObjectTracker:
    def __init__(self, id_manager, ttl_seconds=300.0, iou_threshold=0.15):
        self.id_manager = id_manager
        self.ttl_seconds = ttl_seconds
        self.iou_threshold = iou_threshold
        self.tracks = {}

    def update(self, detected_boxes, face_reid_callback=None, frame=None):
        now = time.time()
        updated_tracks = {}
        unmatched = list(range(len(detected_boxes)))

        candidate_ids = [
            tid for tid, info in self.tracks.items()
            if (now - info["last_seen"]) <= 2.0
        ]

        # Matching por IoU + distância de centro (score ponderado)
        match_candidates = []
        for box_idx in unmatched:
            box = detected_boxes[box_idx]["box"] if isinstance(detected_boxes[box_idx], dict) else detected_boxes[box_idx]
            cx1 = (box[0] + box[2]) / 2.0
            cy1 = (box[1] + box[3]) / 2.0
            h1 = abs(box[3] - box[1])
            for tid in candidate_ids:
                prev_box = self.tracks[tid]["box"]
                iou = compute_iou(box, prev_box)
                cx2 = (prev_box[0] + prev_box[2]) / 2.0
                cy2 = (prev_box[1] + prev_box[3]) / 2.0
                dist = math.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)
                if iou >= self.iou_threshold or (h1 > 0 and (dist / h1) < 0.40):
                    score = iou + max(0, 1.0 - (dist / (h1 + 1e-5)))
                    match_candidates.append((score, tid, box_idx))

        match_candidates.sort(key=lambda x: x[0], reverse=True)
        used_tids = set()
        matched_boxes = set()

        for _, tid, box_idx in match_candidates:
            if tid in used_tids or box_idx in matched_boxes:
                continue
            used_tids.add(tid)
            matched_boxes.add(box_idx)

            box = detected_boxes[box_idx]["box"] if isinstance(detected_boxes[box_idx], dict) else detected_boxes[box_idx]
            prev = self.tracks[tid]
            prev_face_at = prev.get("face_detected_at", 0.0)
            face_fresh = (now - prev_face_at) <= FACE_BOX_TTL

            updated_tracks[tid] = {
                "box":              box,
                "last_seen":        now,
                "triggered":        prev.get("triggered", False),
                "age":              prev.get("age", 0) + 1,
                "identity":         prev.get("identity"),
                "face_box":         prev.get("face_box") if face_fresh else None,
                "face_detected_at": prev_face_at if face_fresh else 0.0,
                "face_status":      prev.get("face_status"),
                "face_confidence":  prev.get("face_confidence"),
                "last_face_check":  prev.get("last_face_check", 0.0),
            }

        # Boxes sem match: tenta Re-ID facial, senão atribui novo ID
        unmatched = [i for i in unmatched if i not in matched_boxes]
        for box_idx in unmatched:
            box = detected_boxes[box_idx]["box"] if isinstance(detected_boxes[box_idx], dict) else detected_boxes[box_idx]
            assigned_id = None
            reidentified = False

            if face_reid_callback and frame is not None:
                x1, y1, x2, y2 = box
                crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                if crop.size > 0:
                    matched = face_reid_callback(crop)
                    if matched is not None and matched not in updated_tracks:
                        assigned_id = matched
                        reidentified = True
                        sys_logger.info(f"[Tracker] Reidentificado via FaceReID: ID #{assigned_id}")

            if assigned_id is None:
                assigned_id = self.id_manager.get_next_id()

            prev = self.tracks.get(assigned_id, {})
            updated_tracks[assigned_id] = {
                "box":              box,
                "last_seen":        now,
                "triggered":        False,
                "age":              prev.get("age", 0) + 1,
                "identity":         assigned_id if reidentified else prev.get("identity"),
                "face_box":         None,
                "face_detected_at": 0.0,
                "face_status":      "identificado" if reidentified else "novo",
                "face_confidence":  prev.get("face_confidence"),
                "last_face_check":  0.0,
            }

        self.tracks.update(updated_tracks)

        # Expira tracks inativos (5 min)
        for tid in [t for t, i in self.tracks.items()
                    if not i.get("triggered", False) and (now - i["last_seen"]) > self.ttl_seconds]:
            sys_logger.info(f"[Tracker] Track #{tid} expirou.")
            self.tracks.pop(tid, None)

        return {
            tid: info for tid, info in self.tracks.items()
            if (now - info["last_seen"]) <= 1.0
        }

    def set_trigger(self, track_id: int):
        if track_id in self.tracks:
            self.tracks[track_id]["triggered"] = True
            sys_logger.info(f"[Tracker] Gatilho mantido para Track #{track_id}.")
