"""
FaceCapture — captura e insights faciais por track.

Mudanças em relação à versão anterior:
  - O FaceDetector não é mais instanciado aqui.
    Recebe o detector injetado (shared) para evitar dois modelos em memória.
  - Throttle por track_id: cada track só passa pelo detector após FACE_CHECK_INTERVAL segundos.
    Evita rodar reconhecimento facial em todos os frames para todas as pessoas.
  - Quando o intervalo não expirou, retorna o último resultado em cache (sem re-detectar).
"""

import time
import cv2
import numpy as np
from utils.logger import sys_logger

# Intervalo mínimo entre detecções faciais para o mesmo track (segundos)
FACE_CHECK_INTERVAL = 0.5


class FaceCapture:
    def __init__(self, face_detector=None):
        """
        Parameters
        ----------
        face_detector : FaceDetector | None
            Instância compartilhada do FaceDetector.
            Se None, cria uma internamente (compatibilidade retroativa).
        """
        if face_detector is not None:
            self.detector = face_detector
            self._owns_detector = False
        else:
            # Fallback: cria o próprio detector (compatibilidade com código legado)
            from detection.face_detector import FaceDetector
            self.detector = FaceDetector()
            self._owns_detector = True

        # Throttle: track_id -> (last_check_timestamp, last_result)
        self._cache: dict[int, tuple[float, dict | None]] = {}

    # ------------------------------------------------------------------
    # API principal
    # ------------------------------------------------------------------

    def capture_face_insights(self, image, track_id: int = -1):
        """
        Detecta rosto em 'image' e retorna insights + embedding.

        Se track_id >= 0 e o intervalo ainda não expirou, retorna o resultado
        em cache sem re-detectar.

        Returns
        -------
        dict | None  com chaves: box, face_img, embedding, insights
        """
        if image is None or image.size == 0:
            return None

        now = time.time()

        # --- Throttle ---
        if track_id >= 0:
            last_ts, last_result = self._cache.get(track_id, (0.0, None))
            if (now - last_ts) < FACE_CHECK_INTERVAL:
                return last_result  # retorna cache, sem re-detectar

        # --- Detecção real ---
        faces = self.detector.detect_faces(image)
        if not faces:
            result = None
        else:
            best = faces[0]  # maior rosto (já ordenado por área no detector)
            face_img = best["face_img"]

            gray = (
                cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                if len(face_img.shape) == 3
                else face_img
            )
            brightness = float(np.mean(gray))
            clarity = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            result = {
                "box": best["box"],
                "face_img": face_img,
                "embedding": best["embedding"],
                "insights": {
                    "brightness": round(brightness, 2),
                    "clarity": round(clarity, 2),
                    "dimensions": [face_img.shape[1], face_img.shape[0]],
                },
            }
            sys_logger.debug(
                f"[FaceCapture] track={track_id} brilho={brightness:.0f} nitidez={clarity:.0f}"
            )

        # --- Atualiza cache ---
        if track_id >= 0:
            self._cache[track_id] = (now, result)

        return result

    def prune_cache(self, active_track_ids: set):
        """Remove entradas de tracks que não existem mais (evita vazamento de memória)."""
        stale = [tid for tid in self._cache if tid not in active_track_ids]
        for tid in stale:
            del self._cache[tid]

    # ------------------------------------------------------------------
    # Compatibilidade
    # ------------------------------------------------------------------

    def close(self):
        if self._owns_detector:
            self.detector.close()
