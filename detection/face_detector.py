import cv2
import numpy as np
import os
import mediapipe as mp
from utils.logger import sys_logger

class FaceDetector:
    def __init__(self):
        self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.use_mediapipe = False
        self.face_landmarker = None

        model_path = "face_landmarker.task"
        if os.path.exists(model_path):
            try:
                base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
                options = mp.tasks.vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    running_mode=mp.tasks.vision.RunningMode.IMAGE,
                    num_faces=5
                )
                self.face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
                self.use_mediapipe = True
                sys_logger.info("MediaPipe FaceLandmarker (468 landmarks Mesh) ativado com sucesso!")
            except Exception as e:
                sys_logger.warning(f"Erro ao inicializar MediaPipe FaceLandmarker: {e}. Usando Haar Cascade.")
        else:
            sys_logger.warning("Modelo face_landmarker.task não encontrado. Usando Haar Cascade.")

    def detect_faces(self, frame):
        """Retorna lista de dicionários com 'box' [x, y, w, h], 'face_img', 'embedding' e 'landmarks'."""
        if frame is None or frame.size == 0:
            return []

        faces_data = []
        h, w = frame.shape[:2]

        if self.use_mediapipe and self.face_landmarker:
            try:
                # MediaPipe requer RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = self.face_landmarker.detect(mp_img)

                if result.face_landmarks:
                    for landmarks in result.face_landmarks:
                        xs = [lm.x * w for lm in landmarks]
                        ys = [lm.y * h for lm in landmarks]
                        x_min, y_min = max(0, int(min(xs))), max(0, int(min(ys)))
                        x_max, y_max = min(w, int(max(xs))), min(h, int(max(ys)))
                        bw, bh = x_max - x_min, y_max - y_min

                        if bw > 10 and bh > 10:
                            face_img = frame[y_min:y_max, x_min:x_max]
                            # Embedding de 468 pontos faciais normalizados
                            coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
                            norm = np.linalg.norm(coords)
                            embedding = coords / norm if norm > 0 else coords

                            faces_data.append({
                                "box": [x_min, y_min, bw, bh],
                                "face_img": face_img,
                                "embedding": embedding,
                                "landmarks": landmarks
                            })
                    return faces_data
            except Exception as e:
                sys_logger.error(f"Erro no MediaPipe FaceLandmarker: {e}")

        # Fallback ultra-robusto com OpenCV Haar Cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (fx, fy, fw, fh) in faces:
            face_crop = frame[fy:fy+fh, fx:fx+fw]
            embedding = self.extract_feature_vector(face_crop)
            faces_data.append({
                "box": [fx, fy, fw, fh],
                "face_img": face_crop,
                "embedding": embedding,
                "landmarks": None
            })

        return faces_data

    def extract_feature_vector(self, face_crop):
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
                cell_mag = mag[i*16:(i+1)*16, j*16:(j+1)*16]
                cell_ang = angle[i*16:(i+1)*16, j*16:(j+1)*16]
                hist, _ = np.histogram(cell_ang, bins=8, range=(0, 360), weights=cell_mag)
                cells.extend(hist)

        vec = np.array(cells, dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def close(self):
        if self.face_landmarker:
            self.face_landmarker.close()
