"""
FaceCapture — captura e insights faciais por track com throttle.

Cada track é verificado no máximo uma vez a cada FACE_CHECK_INTERVAL segundos.
Entre verificações, retorna o último resultado em cache sem re-detectar.
"""

import time
import cv2
import numpy as np
from utils.logger import sys_logger

FACE_CHECK_INTERVAL = 0.5  # segundos entre detecções para o mesmo track


class FaceCapture:
    def __init__(self, face_detector=None):
        if face_detector is not None:
            self.detector       = face_detector
            self._owns_detector = False
        else:
            from detection.face_detector import FaceDetector
            self.detector       = FaceDetector()
            self._owns_detector = True

        # track_id -> (last_check_timestamp, last_result)
        self._cache: dict = {}

    def capture_face_insights(self, image, track_id: int = -1):
        """
        Detecta rosto em 'image' e retorna dict com: box, face_img, embedding, insights.
        Retorna None se nenhum rosto for encontrado.
        Respeita throttle por track_id quando track_id >= 0.
        """
        if image is None or image.size == 0:
            return None

        now = time.time()
        if track_id >= 0:
            last_ts, last_result = self._cache.get(track_id, (0.0, None))
            if (now - last_ts) < FACE_CHECK_INTERVAL:
                return last_result

        faces = self.detector.detect_faces(image)
        if not faces:
            result = None
        else:
            best     = faces[0]
            face_img = best["face_img"]
            gray = (
                cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                if len(face_img.shape) == 3 else face_img
            )
            result = {
                "box":      best["box"],
                "face_img": face_img,
                "embedding": best["embedding"],
                "insights": {
                    "brightness": round(float(np.mean(gray)), 2),
                    "clarity":    round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
                    "dimensions": [face_img.shape[1], face_img.shape[0]],
                },
            }
            sys_logger.debug(
                f"[FaceCapture] track={track_id} "
                f"brilho={result['insights']['brightness']:.0f} "
                f"nitidez={result['insights']['clarity']:.0f}"
            )

        if track_id >= 0:
            self._cache[track_id] = (now, result)

        return result

    def prune_cache(self, active_track_ids: set):
        for tid in [t for t in self._cache if t not in active_track_ids]:
            del self._cache[tid]

    def close(self):
        if self._owns_detector:
            self.detector.close()
