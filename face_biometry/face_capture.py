import cv2
import numpy as np
from detection.face_detector import FaceDetector
from utils.logger import sys_logger

class FaceCapture:
    def __init__(self):
        self.detector = FaceDetector()

    def capture_face_insights(self, image):
        """
        Detecta rosto na imagem dada, extrai a imagem do rosto, vetor de características e insights (brilho e nitidez).
        """
        if image is None or image.size == 0:
            return None

        faces = self.detector.detect_faces(image)
        if not faces:
            return None

        # Pega a primeira/melhor face detectada
        best_face = faces[0]
        face_img = best_face["face_img"]

        # Calcula insights do rosto
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
        brightness = float(np.mean(gray))
        clarity = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        insights = {
            "brightness": round(brightness, 2),
            "clarity": round(clarity, 2),
            "dimensions": [face_img.shape[1], face_img.shape[0]]
        }

        sys_logger.debug(f"[FaceCapture] Rosto capturado com brilho={brightness:.1f}, nitidez={clarity:.1f}")

        return {
            "box": best_face["box"],
            "face_img": face_img,
            "embedding": best_face["embedding"],
            "insights": insights
        }

    def close(self):
        self.detector.close()
