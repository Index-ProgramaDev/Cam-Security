import numpy as np
from utils.logger import sys_logger

class FaceStorage:
    """
    Armazena embeddings faciais na memória organizados por ID.
    Consome IDs exclusivamente fornecidos pelo IDManager.
    """
    def __init__(self):
        self._storage = {}  # track_id (int) -> embedding (np.ndarray)

    def save_embedding(self, track_id: int, embedding: np.ndarray):
        """Salva ou atualiza o vetor facial do ID especificado."""
        if embedding is None or len(embedding) == 0:
            return
        self._storage[track_id] = embedding
        sys_logger.debug(f"[FaceStorage] Embedding facial armazenado para ID #{track_id}")

    def get_embedding(self, track_id: int) -> np.ndarray:
        return self._storage.get(track_id)

    def get_all(self):
        return dict(self._storage)

    def remove(self, track_id: int):
        self._storage.pop(track_id, None)

    def clear(self):
        self._storage.clear()
        sys_logger.info("[FaceStorage] Armazenamento facial em memória limpo.")
