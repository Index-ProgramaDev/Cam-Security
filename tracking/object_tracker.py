import time
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
    def __init__(self, id_manager, ttl_seconds=300.0, iou_threshold=0.3):
        self.id_manager = id_manager
        self.ttl_seconds = ttl_seconds
        self.iou_threshold = iou_threshold
        self.tracks = {}  # track_id -> {"box": [x1, y1, x2, y2], "last_seen": timestamp, "triggered": bool}

    def update(self, detected_boxes, face_reid_callback=None, frame=None):
        """
        Associa detecções do frame atual aos tracks ativos via IoU.
        Retorna apenas os tracks ATIVOS (vistos no frame atual ou há no máximo 1 segundo).
        Retém tracks na memória por 5 minutos (300s) para reidentificação.
        """
        now = time.time()
        updated_tracks = {}
        unmatched_boxes = []

        # Tentar IoU matching com tracks recentemente vistos (< 2 segundos)
        active_candidate_ids = [
            tid for tid, info in self.tracks.items()
            if (now - info["last_seen"]) <= 2.0
        ]
        used_track_ids = set()

        for box_info in detected_boxes:
            box = box_info["box"] if isinstance(box_info, dict) else box_info
            best_iou = 0.0
            best_id = None

            for tid in active_candidate_ids:
                if tid in used_track_ids:
                    continue
                prev_box = self.tracks[tid]["box"]
                iou = compute_iou(box, prev_box)
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_id = tid

            if best_id is not None:
                used_track_ids.add(best_id)
                updated_tracks[best_id] = {
                    "box": box,
                    "last_seen": now,
                    "triggered": self.tracks[best_id].get("triggered", False)
                }
            else:
                unmatched_boxes.append(box)

        # Para caixas não associadas via IoU, tenta FaceReID contra a memória
        for box in unmatched_boxes:
            assigned_id = None

            if face_reid_callback and frame is not None:
                x1, y1, x2, y2 = box
                person_crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                if person_crop.size > 0:
                    matched_reid = face_reid_callback(person_crop)
                    if matched_reid is not None:
                        assigned_id = matched_reid
                        sys_logger.info(f"[Tracker] Pessoa reidentificada! ID #{assigned_id}")

            if assigned_id is None:
                assigned_id = self.id_manager.get_next_id()

            updated_tracks[assigned_id] = {
                "box": box,
                "last_seen": now,
                "triggered": False
            }

        # Atualiza a memória de longo prazo do tracker
        self.tracks.update(updated_tracks)

        # Remove da memória de longo prazo tracks mais antigos que 5 minutos (300s)
        expired_ids = [
            tid for tid, tinfo in self.tracks.items()
            if not tinfo.get("triggered", False) and (now - tinfo["last_seen"]) > self.ttl_seconds
        ]
        for tid in expired_ids:
            sys_logger.info(f"[Tracker] Track #{tid} expirou após 5 minutos.")
            self.tracks.pop(tid, None)

        # Retorna apenas tracks ATIVOS no momento (vistos há menos de 1.0s) para renderização na tela
        currently_visible_tracks = {
            tid: info for tid, info in self.tracks.items()
            if (now - info["last_seen"]) <= 1.0
        }

        return currently_visible_tracks

    def set_trigger(self, track_id: int):
        if track_id in self.tracks:
            self.tracks[track_id]["triggered"] = True
            sys_logger.info(f"[Tracker] Gatilho mantido para Track #{track_id}.")