"""
FaceSnapshotBuffer — buffer facial temporário por track.

Fluxo:
  Track detectado + rosto visível
      → embedding aceito a cada EMBEDDING_CAPTURE_INTERVAL segundos, até MAX_EMBEDDINGS_PER_TRACK
      → foto salva a cada SNAPSHOT_INTERVAL segundos (storage/face_tracks/temp/track_<id>/)
  Após TRACK_TTL_SECONDS sem gatilho → pasta e embeddings descartados silenciosamente
  Gatilho de evento → pasta promovida para storage/face_tracks/events/<event_id>/
                    → embeddings exportados como JSON na mesma pasta

Cada track possui estado INDEPENDENTE de throttle e contagem.
"""

import os
import json
import time
import threading
import shutil
from typing import Dict, List, Optional

import cv2
import numpy as np

from utils.logger import sys_logger
from face_biometry.face_config import (
    EMBEDDING_DIM,
    EMBEDDING_CAPTURE_INTERVAL,
    MAX_EMBEDDINGS_PER_TRACK,
)

# Intervalo entre fotos do rosto (independente do intervalo de embedding)
SNAPSHOT_INTERVAL  = 2.0
TRACK_TTL_SECONDS  = 900.0   # 15 minutos
TEMP_BASE_DIR      = os.path.join("storage", "face_tracks", "temp")
EVENT_BASE_DIR     = os.path.join("storage", "face_tracks", "events")
CLEANUP_INTERVAL   = 60.0


class _TrackBuffer:
    """Estado facial independente por track."""
    __slots__ = (
        "track_id", "temp_dir",
        "last_embedding_ts", "embedding_count",
        "last_snapshot_ts", "last_active_ts",
        "embeddings", "snapshot_count", "lock",
    )

    def __init__(self, track_id: int, temp_dir: str):
        self.track_id          = track_id
        self.temp_dir          = temp_dir
        self.last_embedding_ts = 0.0   # último embedding aceito
        self.embedding_count   = 0     # total aceitos neste buffer
        self.last_snapshot_ts  = 0.0   # última foto salva
        self.last_active_ts    = time.time()
        self.embeddings: List[Dict] = []
        self.snapshot_count    = 0
        self.lock              = threading.Lock()


class FaceSnapshotBuffer:
    """
    Gerencia buffers temporários de fotos e embeddings faciais por track.
    Thread-safe. Estado de throttle e contagem é INDEPENDENTE por track.
    """

    def __init__(self):
        self._tracks: Dict[int, _TrackBuffer] = {}
        self._lock            = threading.Lock()
        self._last_cleanup_ts = time.time()
        os.makedirs(TEMP_BASE_DIR, exist_ok=True)
        os.makedirs(EVENT_BASE_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def on_face_detected(self, track_id: int, face_img: np.ndarray,
                         embedding: np.ndarray, now: Optional[float] = None) -> bool:
        """
        Chamado quando um rosto é detectado para um track.

        - Valida dimensão do embedding contra EMBEDDING_DIM.
        - Aceita embedding apenas se EMBEDDING_CAPTURE_INTERVAL passou para ESTE track.
        - Para após MAX_EMBEDDINGS_PER_TRACK embeddings para este track.
        - Salva foto com throttle de SNAPSHOT_INTERVAL (independente do embedding).

        Retorna True se o embedding foi aceito, False caso contrário.
        """
        now = now or time.time()

        # Validação de dimensão — contrato rígido
        if embedding is None or len(embedding) != EMBEDDING_DIM:
            sys_logger.warning(
                f"[FaceEmbedding] Embedding rejeitado: "
                f"dimensão={len(embedding) if embedding is not None else 'None'}, "
                f"esperado={EMBEDDING_DIM}, track={track_id}"
            )
            return False

        buf = self._get_or_create(track_id)
        embedding_accepted = False

        with buf.lock:
            buf.last_active_ts = now

            # Throttle de embedding por track
            if buf.embedding_count >= MAX_EMBEDDINGS_PER_TRACK:
                sys_logger.debug(
                    f"[FaceEmbedding] Track #{track_id} limite atingido "
                    f"({buf.embedding_count}/{MAX_EMBEDDINGS_PER_TRACK})"
                )
            elif (now - buf.last_embedding_ts) < EMBEDDING_CAPTURE_INTERVAL:
                sys_logger.debug(
                    f"[FaceEmbedding] Track #{track_id} embedding ignorado — "
                    f"intervalo não atingido "
                    f"({now - buf.last_embedding_ts:.1f}s < {EMBEDDING_CAPTURE_INTERVAL}s)"
                )
            else:
                # Aceita embedding
                buf.embeddings.append({
                    "ts":  round(now, 3),
                    "dim": EMBEDDING_DIM,
                    "vec": embedding.tolist(),
                })
                buf.embedding_count   += 1
                buf.last_embedding_ts  = now
                embedding_accepted     = True
                sys_logger.debug(
                    f"[FaceEmbedding] Track #{track_id} embedding capturado "
                    f"({buf.embedding_count}/{MAX_EMBEDDINGS_PER_TRACK})"
                )

            # Foto do rosto — throttle independente do embedding
            should_snap = (
                face_img is not None
                and face_img.size > 0
                and (now - buf.last_snapshot_ts) >= SNAPSHOT_INTERVAL
            )
            if should_snap:
                buf.last_snapshot_ts = now
                buf.snapshot_count  += 1
                fname = f"face_{buf.snapshot_count:04d}_{int(now)}.jpg"
                fpath = os.path.join(buf.temp_dir, fname)
            else:
                fpath = None

        # Salva foto fora do lock
        if fpath:
            try:
                cv2.imwrite(fpath, face_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                sys_logger.debug(f"[FaceSnapshot] Track #{track_id} → {fname}")
            except Exception as e:
                sys_logger.warning(f"[FaceSnapshot] Falha ao salvar foto track #{track_id}: {e}")

        return embedding_accepted

    def on_trigger(self, track_id: int, event_id: str) -> Optional[str]:
        """
        Gatilho de evento: promove pasta temporária do track para events/<event_id>/.
        Exporta embeddings como JSON.
        Retorna o caminho da pasta de destino ou None.
        """
        with self._lock:
            buf = self._tracks.get(track_id)

        if buf is None:
            sys_logger.debug(f"[FaceSnapshot] Gatilho para track #{track_id} sem buffer facial.")
            return None

        with buf.lock:
            src_dir        = buf.temp_dir
            embeddings     = list(buf.embeddings)
            snapshot_count = buf.snapshot_count

        if not os.path.isdir(src_dir):
            return None

        dst_dir = os.path.join(EVENT_BASE_DIR, event_id, f"track_{track_id}")
        try:
            shutil.copytree(src_dir, dst_dir)
            emb_path = os.path.join(dst_dir, "embeddings.json")
            with open(emb_path, "w", encoding="utf-8") as f:
                json.dump({
                    "track_id":    track_id,
                    "event_id":    event_id,
                    "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "embedding_dim": EMBEDDING_DIM,
                    "count":       len(embeddings),
                    "embeddings":  embeddings,
                }, f, ensure_ascii=False, indent=2)
            sys_logger.info(
                f"[FaceStorage] Track #{track_id} promovido → {dst_dir} "
                f"({len(embeddings)} embeddings, {snapshot_count} foto(s))"
            )
            return dst_dir
        except Exception as e:
            sys_logger.error(f"[FaceSnapshot] Falha ao promover track #{track_id}: {e}")
            return None

    def get_embeddings(self, track_id: int) -> List[np.ndarray]:
        """Retorna os embeddings acumulados para um track como lista de ndarrays."""
        with self._lock:
            buf = self._tracks.get(track_id)
        if buf is None:
            return []
        with buf.lock:
            return [np.array(e["vec"], dtype=np.float32) for e in buf.embeddings]

    def get_embedding_count(self, track_id: int) -> int:
        with self._lock:
            buf = self._tracks.get(track_id)
        if buf is None:
            return 0
        with buf.lock:
            return buf.embedding_count

    def prune_inactive(self, active_track_ids: set, now: Optional[float] = None):
        """Remove da memória tracks que não estão mais ativos."""
        with self._lock:
            for tid in [t for t in self._tracks if t not in active_track_ids]:
                del self._tracks[tid]

    def run_cleanup_if_due(self, now: Optional[float] = None):
        """Verifica TTL de pastas temporárias e deleta as expiradas."""
        now = now or time.time()
        if (now - self._last_cleanup_ts) < CLEANUP_INTERVAL:
            return
        self._last_cleanup_ts = now
        self._cleanup_expired(now)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_or_create(self, track_id: int) -> _TrackBuffer:
        with self._lock:
            buf = self._tracks.get(track_id)
            if buf is None:
                temp_dir = os.path.join(TEMP_BASE_DIR, f"track_{track_id}")
                os.makedirs(temp_dir, exist_ok=True)
                buf = _TrackBuffer(track_id, temp_dir)
                self._tracks[track_id] = buf
                sys_logger.debug(f"[FaceSnapshot] Buffer criado para track #{track_id}")
            return buf

    def _cleanup_expired(self, now: float):
        """Deleta pastas temp inativas há mais de TRACK_TTL_SECONDS."""
        if not os.path.isdir(TEMP_BASE_DIR):
            return
        removed = 0
        for entry in os.scandir(TEMP_BASE_DIR):
            if not entry.is_dir():
                continue
            try:
                age = now - entry.stat().st_mtime
            except OSError:
                continue
            if age > TRACK_TTL_SECONDS:
                try:
                    shutil.rmtree(entry.path, ignore_errors=True)
                    removed += 1
                    sys_logger.debug(f"[FaceSnapshot] Pasta expirada removida: {entry.name}")
                except Exception as e:
                    sys_logger.warning(f"[FaceSnapshot] Falha ao remover {entry.path}: {e}")
        if removed:
            sys_logger.info(f"[FaceSnapshot] {removed} pasta(s) facial(is) temporária(s) expirada(s).")
