import time
import uuid
import threading
import numpy as np
from typing import Optional, Dict, List
from utils.logger import sys_logger

TEMP_FACE_TTL_SECONDS = 900   # 15 min sem uso
MIN_QUALITY_SCORE     = 0.15
MAX_SAMPLES_PER_PERSON = 20


class FaceRecord:
    __slots__ = ("face_id", "person_id", "embedding", "quality", "created_at")

    def __init__(self, face_id: str, person_id: str, embedding: np.ndarray, quality: float = 1.0):
        self.face_id    = face_id
        self.person_id  = person_id
        self.embedding  = embedding
        self.quality    = quality
        self.created_at = time.time()


class TemporaryFace:
    __slots__ = ("face_id", "embedding", "quality", "created_at", "last_used", "track_id")

    def __init__(self, face_id: str, embedding: np.ndarray, quality: float, track_id: int):
        self.face_id    = face_id
        self.embedding  = embedding
        self.quality    = quality
        self.created_at = time.time()
        self.last_used  = time.time()
        self.track_id   = track_id


class FaceStorage:
    def __init__(self):
        self._lock                 = threading.RLock()
        self._permanent:           Dict[str, List[FaceRecord]] = {}
        self._temp:                Dict[str, TemporaryFace]    = {}
        self._track_identity_cache: Dict[int, str]             = {}
        self._track_face_map:      Dict[int, str]              = {}

    def save_embedding(self, track_id: int, embedding: np.ndarray, quality: float = 1.0):
        if embedding is None or len(embedding) == 0:
            return
        with self._lock:
            person_id = self._track_identity_cache.get(track_id)
            if person_id:
                self._add_to_person(person_id, embedding, quality, track_id)
                return
            face_id = self._track_face_map.get(track_id)
            if face_id and face_id in self._temp:
                tmp = self._temp[face_id]
                if quality > tmp.quality:
                    tmp.embedding = embedding
                    tmp.quality   = quality
                tmp.last_used = time.time()
            else:
                face_id = str(uuid.uuid4())[:8]
                self._temp[face_id]             = TemporaryFace(face_id, embedding, quality, track_id)
                self._track_face_map[track_id]  = face_id
                sys_logger.debug(f"[FaceStorage] Face temporária criada: {face_id} (track #{track_id})")

    def get_embedding(self, track_id: int) -> Optional[np.ndarray]:
        with self._lock:
            pid = self._track_identity_cache.get(track_id)
            if pid:
                recs = self._permanent.get(pid, [])
                if recs:
                    return max(recs, key=lambda r: r.quality).embedding
            fid = self._track_face_map.get(track_id)
            if fid and fid in self._temp:
                return self._temp[fid].embedding
        return None

    def get_all(self) -> Dict[int, np.ndarray]:
        result = {}
        with self._lock:
            for tid, pid in self._track_identity_cache.items():
                recs = self._permanent.get(pid, [])
                if recs:
                    result[tid] = max(recs, key=lambda r: r.quality).embedding
            for tid, fid in self._track_face_map.items():
                if tid not in result and fid in self._temp:
                    result[tid] = self._temp[fid].embedding
        return result

    def get_all_persons(self) -> Dict[str, List[FaceRecord]]:
        with self._lock:
            return {pid: list(recs) for pid, recs in self._permanent.items()}

    def associate_track_to_person(self, track_id: int, person_id: str):
        with self._lock:
            if self._track_identity_cache.get(track_id) == person_id:
                return
            self._track_identity_cache[track_id] = person_id
            sys_logger.info(f"[FaceStorage] Track #{track_id} → Person '{person_id}'")
            fid = self._track_face_map.get(track_id)
            if fid and fid in self._temp:
                tmp = self._temp[fid]
                if tmp.quality >= MIN_QUALITY_SCORE:
                    self._add_to_person(person_id, tmp.embedding, tmp.quality, track_id, face_id=fid)
                    del self._temp[fid]
                    sys_logger.info(f"[FaceStorage] Face temporária {fid} promovida → '{person_id}'")

    def promote_temp_face(self, track_id: int, person_id: str) -> Optional[str]:
        with self._lock:
            fid = self._track_face_map.get(track_id)
            if not fid or fid not in self._temp:
                return None
            tmp = self._temp[fid]
            if tmp.quality < MIN_QUALITY_SCORE:
                sys_logger.warning(f"[FaceStorage] Face {fid} qualidade baixa ({tmp.quality:.2f}) — mantida sem promoção.")
                return fid
            self.associate_track_to_person(track_id, person_id)
            return fid

    def get_identity_for_track(self, track_id: int) -> Optional[str]:
        with self._lock:
            return self._track_identity_cache.get(track_id)

    def get_face_id_for_track(self, track_id: int) -> Optional[str]:
        with self._lock:
            return self._track_face_map.get(track_id)

    def expire_temp_faces(self) -> List[str]:
        now, expired = time.time(), []
        with self._lock:
            for fid, tmp in list(self._temp.items()):
                if (now - tmp.last_used) > TEMP_FACE_TTL_SECONDS:
                    del self._temp[fid]
                    expired.append(fid)
                    for t in [t for t, f in self._track_face_map.items() if f == fid]:
                        del self._track_face_map[t]
                    sys_logger.debug(f"[FaceStorage] Face temporária expirada: {fid}")
        if expired:
            sys_logger.info(f"[FaceStorage] {len(expired)} face(s) temporária(s) expirada(s).")
        return expired

    def register_person(self, person_id: str, embedding: np.ndarray, quality: float = 1.0):
        if embedding is None or len(embedding) == 0:
            return
        with self._lock:
            self._add_to_person(person_id, embedding, quality)
        sys_logger.info(f"[FaceStorage] Embedding registrado para '{person_id}'")

    def remove(self, track_id: int):
        with self._lock:
            self._track_identity_cache.pop(track_id, None)
            self._track_face_map.pop(track_id, None)

    def clear(self):
        with self._lock:
            self._permanent.clear()
            self._temp.clear()
            self._track_identity_cache.clear()
            self._track_face_map.clear()
        sys_logger.info("[FaceStorage] Armazenamento limpo.")

    def _add_to_person(self, person_id: str, embedding: np.ndarray,
                       quality: float, track_id: int = -1, face_id: Optional[str] = None):
        if quality < MIN_QUALITY_SCORE:
            return
        rec = FaceRecord(face_id or str(uuid.uuid4())[:8], person_id, embedding.copy(), quality)
        self._permanent.setdefault(person_id, []).append(rec)
        if len(self._permanent[person_id]) > MAX_SAMPLES_PER_PERSON:
            self._permanent[person_id].sort(key=lambda r: r.quality)
            self._permanent[person_id].pop(0)
