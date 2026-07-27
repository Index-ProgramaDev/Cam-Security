import numpy as np
from utils.logger import sys_logger

FACE_MATCH_THRESHOLD = 0.6  # Limiar ajustável para similaridade de cosseno

class FaceReID:
    """
    Realiza a reidentificação facial comparando vetores com o FaceStorage em memória.
    """
    def __init__(self, face_storage, threshold: float = FACE_MATCH_THRESHOLD):
        self.face_storage = face_storage
        self.threshold = threshold

    def match_embedding(self, query_embedding: np.ndarray):
        """
        Retorna o track_id correspondente caso a similaridade seja superior ao threshold.
        """
        if query_embedding is None or len(query_embedding) == 0:
            return None

        stored_embeddings = self.face_storage.get_all()
        if not stored_embeddings:
            return None

        best_id = None
        best_similarity = -1.0

        q_norm = np.linalg.norm(query_embedding)
        if q_norm == 0:
            return None

        for track_id, stored_vec in stored_embeddings.items():
            s_norm = np.linalg.norm(stored_vec)
            if s_norm == 0:
                continue

            similarity = float(np.dot(query_embedding, stored_vec) / (q_norm * s_norm))

            if similarity > best_similarity:
                best_similarity = similarity
                best_id = track_id

        if best_similarity >= self.threshold:
            sys_logger.info(f"[FaceReID] Match encontrado! ID #{best_id} (Similaridade={best_similarity:.2f})")
            return best_id

        return None
