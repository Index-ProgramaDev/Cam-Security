"""
FaceDetector — versão otimizada para o pipeline CamSecurity.

O objetivo deste módulo é:
  1. Localizar rostos em um crop de pessoa.
  2. Retornar a bounding box e a imagem do rosto.
  3. Gerar um embedding HOG compacto (128-dim) para re-identificação.

O MediaPipe FaceLandmarker (468 landmarks + blendshapes) foi removido porque:
  - Calculava centenas de landmarks usados apenas para estimar a bbox do rosto.
  - Era carregado duas vezes (FaceDetector + FaceCapture) desperdiçando memória/CPU.
  - O embedding de identidade é gerado pelo extract_feature_vector() (HOG), não pelos landmarks.

Substituto: OpenCV Haar Cascade (haarcascade_frontalface_default.xml)
  - Leve, sem dependência de .task files externos.
  - Adequado para localização + crop.
  - Embedding HOG 128-dim para re-identificação via FaceReID.

Throttle por track: o caller (FaceCapture) controla a frequência de chamada.
Este módulo não mantém estado de intervalo — apenas detecta quando chamado.
"""

import cv2
import numpy as np
from utils.logger import sys_logger


# Tamanho mínimo de rosto para aceitar a detecção (pixels)
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

    # ------------------------------------------------------------------
    # Detecção pública
    # ------------------------------------------------------------------

    def detect_faces(self, frame):
        """
        Detecta rostos em 'frame' (pode ser o frame inteiro ou um crop de pessoa).

        Retorna lista de dicts:
          {
            "box":       [x, y, w, h],   # coordenadas no espaço de 'frame'
            "face_img":  np.ndarray,
            "embedding": np.ndarray,     # vetor HOG 128-dim normalizado
            "landmarks": None,           # mantido para compatibilidade
          }
        """
        if frame is None or frame.size == 0:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_hw = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
        )

        results = []
        if len(faces_hw) == 0:
            return results

        # Ordena por área decrescente → o maior rosto é o principal
        faces_sorted = sorted(faces_hw, key=lambda f: f[2] * f[3], reverse=True)

        h_frame, w_frame = frame.shape[:2]
        for (fx, fy, fw, fh) in faces_sorted:
            # Recorte seguro
            x1 = max(0, fx)
            y1 = max(0, fy)
            x2 = min(w_frame, fx + fw)
            y2 = min(h_frame, fy + fh)
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue
            embedding = self.extract_feature_vector(face_img)
            results.append(
                {
                    "box": [x1, y1, x2 - x1, y2 - y1],
                    "face_img": face_img,
                    "embedding": embedding,
                    "landmarks": None,
                }
            )

        return results

    # ------------------------------------------------------------------
    # Embedding compacto
    # ------------------------------------------------------------------

    def extract_feature_vector(self, face_crop):
        """
        Gera vetor HOG de 128 dimensões normalizado por L2.

        Grade 4×4 de células, 8 bins de orientação por célula → 128 valores.
        Rápido, sem dependências externas, bom para similaridade de cosseno.
        """
        if face_crop is None or face_crop.size == 0:
            return np.zeros(128, dtype=np.float32)

        resized = cv2.resize(face_crop, (64, 64))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

        cells = []
        for i in range(4):
            for j in range(4):
                cell_mag = mag[i * 16 : (i + 1) * 16, j * 16 : (j + 1) * 16]
                cell_ang = angle[i * 16 : (i + 1) * 16, j * 16 : (j + 1) * 16]
                hist, _ = np.histogram(cell_ang, bins=8, range=(0, 360), weights=cell_mag)
                cells.extend(hist)

        vec = np.array(cells, dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    # ------------------------------------------------------------------
    # Compatibilidade
    # ------------------------------------------------------------------

    def close(self):
        """Sem recursos a liberar (Haar Cascade não precisa de cleanup)."""
        pass
