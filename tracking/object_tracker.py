import time
import math
from utils.logger import sys_logger

def compute_iou(boxA, boxB):
    """Calcula Intersection over Union (IoU) entre duas caixas [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

class ObjectTracker:
    def __init__(self, id_manager, ttl_seconds=300.0, iou_threshold=0.15):
        self.id_manager = id_manager
        self.ttl_seconds = ttl_seconds
        self.iou_threshold = iou_threshold
        # tracks: track_id -> {"box": [...], "last_seen": timestamp, "triggered": bool, "age": int}
        self.tracks = {}

    def update(self, detected_boxes, face_reid_callback=None, frame=None):
        now = time.time()
        updated_tracks = {}
        unmatched_box_indices = list(range(len(detected_boxes)))

        # Tentar IoU + Center Distance matching com candidate tracks recentemente vistos (< 2.0s)
        candidate_ids = [
            tid for tid, info in self.tracks.items()
            if (now - info["last_seen"]) <= 2.0
        ]

        # Monta matriz de correspondência com pares (score, tid, box_idx)
        match_candidates = []
        for box_idx in unmatched_box_indices:
            box = detected_boxes[box_idx]["box"] if isinstance(detected_boxes[box_idx], dict) else detected_boxes[box_idx]
            cx1, cy1 = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            h1 = abs(box[3] - box[1])

            for tid in candidate_ids:
                prev_box = self.tracks[tid]["box"]
                iou = compute_iou(box, prev_box)
                cx2, cy2 = (prev_box[0] + prev_box[2]) / 2.0, (prev_box[1] + prev_box[3]) / 2.0
                dist = math.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)

                # Se houver sobreposição IoU >= 0.15 OU distância do centro for pequena (< 40% da altura)
                if iou >= self.iou_threshold or (h1 > 0 and (dist / h1) < 0.40):
                    # Score ponderado: prioriza IoU alto, com bônus de proximidade de centro
                    score = iou + max(0, 1.0 - (dist / (h1 + 1e-5)))
                    match_candidates.append((score, tid, box_idx))

        # Atribuição gulosa pela maior pontuação
        match_candidates.sort(key=lambda x: x[0], reverse=True)
        used_track_ids = set()
        matched_box_indices = set()

        for score, tid, box_idx in match_candidates:
            if tid in used_track_ids or box_idx in matched_box_indices:
                continue

            used_track_ids.add(tid)
            matched_box_indices.add(box_idx)
            box = detected_boxes[box_idx]["box"] if isinstance(detected_boxes[box_idx], dict) else detected_boxes[box_idx]
            prev_age = self.tracks[tid].get("age", 0)

            updated_tracks[tid] = {
                "box": box,
                "last_seen": now,
                "triggered": self.tracks[tid].get("triggered", False),
                "age": prev_age + 1
            }

        # Para caixas não associadas via IoU/Centro, tenta FaceReID contra a memória
        unmatched_box_indices = [i for i in unmatched_box_indices if i not in matched_box_indices]

        for box_idx in unmatched_box_indices:
            box = detected_boxes[box_idx]["box"] if isinstance(detected_boxes[box_idx], dict) else detected_boxes[box_idx]
            assigned_id = None

            if face_reid_callback and frame is not None:
                x1, y1, x2, y2 = box
                person_crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                if person_crop.size > 0:
                    matched_reid = face_reid_callback(person_crop)
                    if matched_reid is not None:
                        assigned_id = matched_reid
                        sys_logger.info(f"[Tracker] Pessoa reidentificada via FaceReID! ID #{assigned_id}")

            if assigned_id is None:
                assigned_id = self.id_manager.get_next_id()

            prev_age = self.tracks.get(assigned_id, {}).get("age", 0)
            updated_tracks[assigned_id] = {
                "box": box,
                "last_seen": now,
                "triggered": False,
                "age": prev_age + 1
            }

        # Atualiza a memória do tracker
        self.tracks.update(updated_tracks)

        # Expira tracks na memória após TTL de 5 min (300s)
        expired_ids = [
            tid for tid, tinfo in self.tracks.items()
            if not tinfo.get("triggered", False) and (now - tinfo["last_seen"]) > self.ttl_seconds
        ]
        for tid in expired_ids:
            sys_logger.info(f"[Tracker] Track #{tid} expirou após 5 minutos.")
            self.tracks.pop(tid, None)

        # Retorna apenas tracks visíveis atualmente (< 1.0s de latência)
        currently_visible_tracks = {
            tid: info for tid, info in self.tracks.items()
            if (now - info["last_seen"]) <= 1.0
        }

        return currently_visible_tracks

    def set_trigger(self, track_id: int):
        if track_id in self.tracks:
            self.tracks[track_id]["triggered"] = True
            sys_logger.info(f"[Tracker] Gatilho mantido para Track #{track_id}.")