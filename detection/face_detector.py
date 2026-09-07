

import cv2
import numpy as np
from utils.logger import sys_logger
from face_biometry.face_config import EMBEDDING_DIM

MIN_FACE_SIZE = 28


class FaceDetector:
    def __init__(self):
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if self.cascade.empty():
            sys_logger.error("[FaceDetector] Haar Cascade não carregou.")
        else:
            sys_logger.info("[FaceDetector] Haar Cascade (frontal) inicializado — modo leve.")

    def detect_faces(self, frame):
    
        if frame is None or frame.size == 0:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_hw = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
        )

        if len(faces_hw) == 0:
            return []

        faces_sorted = sorted(faces_hw, key=lambda f: f[2] * f[3], reverse=True)
        h_frame, w_frame = frame.shape[:2]
        results = []

        for (fx, fy, fw, fh) in faces_sorted:
            x1 = max(0, fx)
            y1 = max(0, fy)
            x2 = min(w_frame, fx + fw)
            y2 = min(h_frame, fy + fh)
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue
            results.append({
                "box":       [x1, y1, x2 - x1, y2 - y1],
                "face_img":  face_img,
                "embedding": self.extract_feature_vector(face_img),
                "landmarks": None,
            })

        return results

    def extract_feature_vector(self, face_crop):
        """
        HOG simplificado: grade 4×4 células × 8 bins = 128 dimensões.
        Dimensão de saída sempre == EMBEDDING_DIM (128).
        """
        if face_crop is None or face_crop.size == 0:
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

        resized = cv2.resize(face_crop, (64, 64))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

        cells = []
        for i in range(4):
            for j in range(4):
                cell_mag = mag[i * 16:(i + 1) * 16, j * 16:(j + 1) * 16]
                cell_ang = angle[i * 16:(i + 1) * 16, j * 16:(j + 1) * 16]
                hist, _ = np.histogram(cell_ang, bins=8, range=(0, 360), weights=cell_mag)
                cells.extend(hist)

        vec = np.array(cells, dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def close(self):
        pass
