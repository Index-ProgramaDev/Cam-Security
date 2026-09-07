"""
TrackRegistry — persistência e reserva de IDs de tracking entre sessões.

Problema resolvido:
  Sem isso, o IDManager reinicia em #1 a cada execução.
  Track #1 da sessão anterior (João) pode ser reatribuído a outra pessoa.

Solução:
  1. Cada track com embedding facial é persistido em JSON ao disco.
  2. Ao reiniciar, o IDManager é informado dos IDs já usados → não reatribui.
  3. O Re-ID existente pode reassociar um track recuperado ao ID original de João.
  4. Após TRACK_TTL_SECONDS sem atividade, o registro é liberado → ID pode ser reutilizado.

Arquivo de estado: storage/track_registry.json
"""

import os
import json
import time
import threading
from typing import Dict, List, Optional

import numpy as np

from utils.logger import sys_logger
from face_biometry.face_config import EMBEDDING_DIM

REGISTRY_PATH      = os.path.join("storage", "track_registry.json")
TRACK_TTL_SECONDS  = 900.0   # 15 minutos — igual ao FaceSnapshotBuffer
SAVE_INTERVAL      = 10.0    # salva no disco no máximo a cada 10s (evita I/O excessivo)
MAX_EMBEDDINGS     = 20      # máximo de embeddings por track no registry


class TrackEntry:
    """Entrada de um track no registry."""

    __slots__ = (
        "track_id", "person_id", "embeddings",
        "first_seen", "last_seen", "triggered",
    )

    def __init__(self, track_id: int, person_id: Optional[str] = None):
        self.track_id   = track_id
        self.person_id  = person_id
        self.embeddings: List[List[float]] = []   # vetores serializados
        self.first_seen = time.time()
        self.last_seen  = time.time()
        self.triggered  = False   # se houve gatilho de alerta — nunca expira sozinho

    def to_dict(self) -> dict:
        return {
            "track_id":   self.track_id,
            "person_id":  self.person_id,
            "embeddings": self.embeddings,
            "first_seen": round(self.first_seen, 3),
            "last_seen":  round(self.last_seen, 3),
            "triggered":  self.triggered,
        }

    @staticmethod
    def from_dict(d: dict) -> "TrackEntry":
        e            = TrackEntry(d["track_id"], d.get("person_id"))
        e.embeddings = d.get("embeddings", [])
        e.first_seen = d.get("first_seen", time.time())
        e.last_seen  = d.get("last_seen", time.time())
        e.triggered  = d.get("triggered", False)
        return e

    def is_expired(self, now: float) -> bool:
        """Expira apenas se não houve gatilho E passou o TTL."""
        if self.triggered:
            return False
        return (now - self.last_seen) > TRACK_TTL_SECONDS

    def best_embedding(self) -> Optional[np.ndarray]:
        """Retorna o último embedding registrado como ndarray."""
        if not self.embeddings:
            return None
        return np.array(self.embeddings[-1], dtype=np.float32)

    def all_embeddings(self) -> List[np.ndarray]:
        return [np.array(v, dtype=np.float32) for v in self.embeddings]


class TrackRegistry:
    """
    Registry persistido de tracks com embeddings faciais.
    Thread-safe.
    """

    def __init__(self, id_manager=None):
        self._entries: Dict[int, TrackEntry] = {}
        self._lock           = threading.RLock()
        self._id_manager     = id_manager
        self._dirty          = False
        self._last_save_ts   = 0.0

        os.makedirs("storage", exist_ok=True)
        self._load()

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------

    def _load(self):
        """Carrega registry do disco e reserva IDs no IDManager."""
        if not os.path.exists(REGISTRY_PATH):
            return
        try:
            with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            now = time.time()
            loaded = expired = 0
            for d in data.get("tracks", []):
                entry = TrackEntry.from_dict(d)
                if entry.is_expired(now):
                    expired += 1
                    continue
                self._entries[entry.track_id] = entry
                if self._id_manager:
                    self._id_manager.register_id(entry.track_id)
                loaded += 1
            sys_logger.info(
                f"[TrackRegistry] Carregado: {loaded} track(s) ativos, "
                f"{expired} expirado(s) descartados."
            )
        except Exception as e:
            sys_logger.warning(f"[TrackRegistry] Falha ao carregar {REGISTRY_PATH}: {e}")

    def save(self, force: bool = False):
        """Persiste registry no disco. Throttled a SAVE_INTERVAL segundos."""
        now = time.time()
        if not force and not self._dirty:
            return
        if not force and (now - self._last_save_ts) < SAVE_INTERVAL:
            return
        with self._lock:
            entries_snapshot = list(self._entries.values())
        try:
            payload = {
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "tracks":   [e.to_dict() for e in entries_snapshot],
            }
            tmp = REGISTRY_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp, REGISTRY_PATH)
            self._dirty        = False
            self._last_save_ts = now
        except Exception as e:
            sys_logger.warning(f"[TrackRegistry] Falha ao salvar: {e}")

    # ------------------------------------------------------------------
    # Operações de track
    # ------------------------------------------------------------------

    def touch(self, track_id: int, now: Optional[float] = None):
        """Atualiza last_seen de um track ativo."""
        now = now or time.time()
        with self._lock:
            entry = self._entries.get(track_id)
            if entry:
                entry.last_seen = now
                self._dirty = True

    def add_embedding(self, track_id: int, embedding: np.ndarray,
                      person_id: Optional[str] = None, now: Optional[float] = None):
        """
        Adiciona embedding ao track. Cria entrada se não existir.
        Rejeita embeddings com dimensão diferente de EMBEDDING_DIM.
        Mantém no máximo MAX_EMBEDDINGS por track.
        """
        if embedding is None or len(embedding) != EMBEDDING_DIM:
            sys_logger.warning(
                f"[TrackRegistry] Embedding rejeitado: "
                f"dimensão={len(embedding) if embedding is not None else 'None'}, "
                f"esperado={EMBEDDING_DIM}, track={track_id}"
            )
            return
        now = now or time.time()
        with self._lock:
            entry = self._entries.get(track_id)
            if entry is None:
                entry = TrackEntry(track_id, person_id)
                self._entries[track_id] = entry
                if self._id_manager:
                    self._id_manager.register_id(track_id)

            entry.last_seen = now
            if person_id and entry.person_id is None:
                entry.person_id = person_id

            vec = embedding.tolist()
            if vec not in entry.embeddings:
                entry.embeddings.append(vec)
                if len(entry.embeddings) > MAX_EMBEDDINGS:
                    entry.embeddings.pop(0)

            self._dirty = True

    def set_person(self, track_id: int, person_id: str):
        """Associa identidade ao track."""
        with self._lock:
            entry = self._entries.get(track_id)
            if entry and entry.person_id != person_id:
                entry.person_id = person_id
                self._dirty = True
                sys_logger.info(f"[TrackRegistry] Track #{track_id} → '{person_id}'")

    def set_triggered(self, track_id: int):
        """Marca que houve gatilho de alerta — track não expira automaticamente."""
        with self._lock:
            entry = self._entries.get(track_id)
            if entry and not entry.triggered:
                entry.triggered = True
                self._dirty = True

    def get_entry(self, track_id: int) -> Optional[TrackEntry]:
        with self._lock:
            return self._entries.get(track_id)

    def find_by_embedding(self, query: np.ndarray,
                          threshold: float = 0.70) -> Optional[int]:
        """
        Busca no registry um track cujo embedding seja similar ao query.
        Retorna o track_id do melhor match ou None.
        Usado na reentrada: pessoa reaparece → associar ao track original.
        """
        if query is None or len(query) == 0:
            return None
        q_norm = np.linalg.norm(query)
        if q_norm < 1e-8:
            return None
        q_unit = query / q_norm

        now = time.time()
        best_id, best_sim = None, -1.0

        with self._lock:
            entries = list(self._entries.items())

        for tid, entry in entries:
            if entry.is_expired(now):
                continue
            for vec in entry.all_embeddings():
                v_norm = np.linalg.norm(vec)
                if v_norm < 1e-8 or len(vec) != len(query):
                    continue
                sim = float(np.dot(q_unit, vec / v_norm))
                if sim > best_sim:
                    best_sim, best_id = sim, tid

        if best_sim >= threshold:
            sys_logger.info(
                f"[TrackRegistry] Reentrada detectada: embedding → Track #{best_id} "
                f"(sim={best_sim:.3f})"
            )
            return best_id
        return None

    def expire_old(self, now: Optional[float] = None) -> List[int]:
        """Remove entradas expiradas. Retorna lista de track_ids removidos."""
        now = now or time.time()
        expired = []
        with self._lock:
            for tid in [t for t, e in self._entries.items() if e.is_expired(now)]:
                del self._entries[tid]
                expired.append(tid)
                self._dirty = True
        if expired:
            sys_logger.info(
                f"[TrackRegistry] {len(expired)} track(s) expirado(s): "
                f"{expired}"
            )
        return expired

    def active_ids(self) -> set:
        """Retorna conjunto de track_ids não expirados."""
        now = time.time()
        with self._lock:
            return {tid for tid, e in self._entries.items() if not e.is_expired(now)}

    def summary(self) -> str:
        with self._lock:
            n = len(self._entries)
            with_person = sum(1 for e in self._entries.values() if e.person_id)
        return f"{n} track(s) no registry ({with_person} identificado(s))"
