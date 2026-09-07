import cv2
import numpy as np
from pathlib import Path
from utils.logger import sys_logger
from face_biometry.face_config import EMBEDDING_DIM

MIN_FACE_SIZE = 28


class FaceDetector:
    def __init__(self):
        model_path = (
            Path(__file__).resolve().parent
            / "face_detection_yunet_2023mar.onnx"
        )
        sys_logger.info(f"[FaceDetector] Carregando YuNet: {model_path}")
        
        # A inicialização inicial tem um input_size genérico, é ajustado dinamicamente no detect_faces
        self.yunet = cv2.FaceDetectorYN.create(
            model=str(model_path),
            config="",
            input_size=(320, 320),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000
        )
        import threading
        self._lock = threading.Lock()
        
        sys_logger.info("[FaceDetector] YuNet inicializado com sucesso.")

    def detect_faces(self, frame):
        if frame is None or frame.size == 0:
            return []

        h_frame, w_frame = frame.shape[:2]
        
        with self._lock:
            self.yunet.setInputSize((w_frame, h_frame))
            _, faces = self.yunet.detect(frame)

        if faces is None:
            return []

        results = []
        for face in faces:
            box = face[0:4].astype(int)
            x1 = max(0, box[0])
            y1 = max(0, box[1])
            fw = max(1, box[2])
            fh = max(1, box[3])
            
            x2 = min(w_frame, x1 + fw)
            y2 = min(h_frame, y1 + fh)

            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue

            results.append({
                "box": [x1, y1, x2 - x1, y2 - y1],
                "face_img": face_img,
                "embedding": self.extract_feature_vector(face_img),
                "landmarks": face[4:14].reshape((5, 2)),
            })
            
        results.sort(key=lambda x: x["box"][2] * x["box"][3], reverse=True)
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

        mag, angle = cv2.cartToPolar(
            gx,
            gy,
            angleInDegrees=True
        )

        cells = []

        for i in range(4):
            for j in range(4):
                cell_mag = mag[
                    i * 16:(i + 1) * 16,
                    j * 16:(j + 1) * 16
                ]

                cell_ang = angle[
                    i * 16:(i + 1) * 16,
                    j * 16:(j + 1) * 16
                ]

                hist, _ = np.histogram(
                    cell_ang,
                    bins=8,
                    range=(0, 360),
                    weights=cell_mag
                )

                cells.extend(hist)

        vec = np.array(cells, dtype=np.float32)
        norm = np.linalg.norm(vec)

        return vec / norm if norm > 0 else vec

    def close(self):
        pass
