import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from utils.logger import sys_logger

MATCH_THRESHOLD           = 0.70
HIGH_CONFIDENCE_THRESHOLD = 0.82
MIN_QUALITY_THRESHOLD     = 0.15
FACE_MATCH_THRESHOLD      = MATCH_THRESHOLD  # compat


@dataclass
class ReIDResult:
    person_id:  Optional[str]
    similarity: float
    confidence: str  # "HIGH_CONFIDENCE" | "MATCH" | "NO_MATCH"
    candidates: List[Tuple[str, float]] = field(default_factory=list)

    def is_match(self) -> bool:
        return self.confidence in ("MATCH", "HIGH_CONFIDENCE")

    def is_high_confidence(self) -> bool:
        return self.confidence == "HIGH_CONFIDENCE"


class FaceReID:
    def __init__(self, face_storage,
                 threshold: float = MATCH_THRESHOLD,
                 high_confidence_threshold: float = HIGH_CONFIDENCE_THRESHOLD):
        self.face_storage              = face_storage
        self.threshold                 = threshold
        self.high_confidence_threshold = high_confidence_threshold
        self._identity_cache: dict     = {}
        self._CACHE_TTL                = 30.0

    def identify_by_embedding(self, query_embedding: np.ndarray,
                               embedding_dim: Optional[int] = None) -> ReIDResult:
        no_match = ReIDResult(person_id=None, similarity=0.0, confidence="NO_MATCH")

        if query_embedding is None or len(query_embedding) == 0:
            return no_match
        q_norm = np.linalg.norm(query_embedding)
        if q_norm < 1e-8:
            return no_match
        if embedding_dim is not None and len(query_embedding) != embedding_dim:
            sys_logger.warning(f"[FaceReID] Dimensão incompatível: query={len(query_embedding)}, esperado={embedding_dim}")
            return no_match

        q_unit  = query_embedding / q_norm
        persons = self.face_storage.get_all_persons()
        if not persons:
            return no_match

        best_person, best_sim, candidates = None, -1.0, []

        for person_id, records in persons.items():
            good = [r for r in records if r.quality >= MIN_QUALITY_THRESHOLD]
            if not good:
                continue
            person_best = max(
                (float(np.dot(q_unit, r.embedding / np.linalg.norm(r.embedding)))
                 for r in good
                 if len(r.embedding) == len(query_embedding) and np.linalg.norm(r.embedding) >= 1e-8),
                default=-1.0,
            )
            if person_best >= self.threshold:
                candidates.append((person_id, person_best))
                if person_best > best_sim:
                    best_sim, best_person = person_best, person_id

        candidates.sort(key=lambda x: x[1], reverse=True)

        if best_person is None:
            return ReIDResult(person_id=None, similarity=max(best_sim, 0.0),
                              confidence="NO_MATCH", candidates=candidates)

        confidence = "HIGH_CONFIDENCE" if best_sim >= self.high_confidence_threshold else "MATCH"
        sys_logger.info(f"[FaceReID] '{best_person}' ({confidence}, sim={best_sim:.3f}) | {len(candidates)} candidato(s)")
        return ReIDResult(person_id=best_person, similarity=best_sim,
                          confidence=confidence, candidates=candidates)

    def identify_track(self, track_id: int, query_embedding: np.ndarray,
                       force: bool = False) -> ReIDResult:
        now    = time.time()
        cached = self._identity_cache.get(track_id)
        if cached and not force and (now - cached[3]) < self._CACHE_TTL:
            pid, sim, conf, _ = cached
            return ReIDResult(person_id=pid, similarity=sim, confidence=conf)

        result = self.identify_by_embedding(query_embedding)
        if result.is_match():
            self._identity_cache[track_id] = (result.person_id, result.similarity, result.confidence, now)
            self.face_storage.associate_track_to_person(track_id, result.person_id)
        return result

    def invalidate_cache(self, track_id: int):
        self._identity_cache.pop(track_id, None)

    def prune_cache(self, active_track_ids: set):
        for t in [t for t in self._identity_cache if t not in active_track_ids]:
            del self._identity_cache[t]

    def match_embedding(self, query_embedding: np.ndarray) -> Optional[int]:
        """Compatibilidade com ObjectTracker — busca por track_id no storage legado."""
        if query_embedding is None or len(query_embedding) == 0:
            return None
        stored = self.face_storage.get_all()
        if not stored:
            return None
        q_norm = np.linalg.norm(query_embedding)
        if q_norm < 1e-8:
            return None
        q_unit = query_embedding / q_norm
        best_id, best_sim = None, -1.0
        for tid, ref in stored.items():
            if ref is None or len(ref) != len(query_embedding):
                continue
            r_norm = np.linalg.norm(ref)
            if r_norm < 1e-8:
                continue
            sim = float(np.dot(q_unit, ref / r_norm))
            if sim > best_sim:
                best_sim, best_id = sim, tid
        if best_sim >= self.threshold:
            sys_logger.info(f"[FaceReID] Match: ID #{best_id} (sim={best_sim:.2f})")
            return best_id
        return None
