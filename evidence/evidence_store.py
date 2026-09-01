import os
import uuid
import time
import json
from typing import Optional
from utils.logger import sys_logger

DB_URL         = os.environ.get("CAM_DB_URL", "")
EVIDENCES_DIR  = os.path.join("storage", "evidences")
FALLBACK_FILE  = os.path.join(EVIDENCES_DIR, "evidences_fallback.jsonl")
_db_available  = False
_Session       = None
_EvidenceModel = None


def _init_db():
    global _db_available, _Session, _EvidenceModel
    if _db_available or not DB_URL:
        return
    try:
        from sqlalchemy import create_engine, Column, String, Integer, BigInteger, DateTime, Text
        from sqlalchemy.orm import declarative_base, sessionmaker
        from sqlalchemy.sql import func

        engine = create_engine(DB_URL, pool_pre_ping=True)
        Base   = declarative_base()

        class Evidence(Base):
            __tablename__ = "evidences"
            id           = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
            event_id     = Column(String(128), nullable=False, index=True)
            type         = Column(String(32), nullable=False, default="video")
            storage_path = Column(Text, nullable=False)
            mime_type    = Column(String(64), default="video/mp4")
            size_bytes   = Column(BigInteger, default=0)
            duration_ms  = Column(Integer, default=0)
            created_at   = Column(DateTime(timezone=True), server_default=func.now())
            track_id     = Column(Integer, nullable=True)
            face_id      = Column(String(64), nullable=True)
            person_id    = Column(String(128), nullable=True)

        Base.metadata.create_all(engine)
        _Session, _EvidenceModel, _db_available = sessionmaker(bind=engine), Evidence, True
        sys_logger.info("[EvidenceStore] Conectado ao PostgreSQL.")
    except Exception as e:
        sys_logger.warning(f"[EvidenceStore] Banco não disponível ({e}). Usando fallback JSON.")


def save_evidence_meta(event_id: str, storage_path: str, mime_type: str = "video/mp4",
                       size_bytes: int = 0, duration_ms: int = 0, evidence_type: str = "video",
                       track_id: Optional[int] = None, face_id: Optional[str] = None,
                       person_id: Optional[str] = None) -> Optional[str]:
    _init_db()
    eid = str(uuid.uuid4())

    if _db_available and _Session and _EvidenceModel:
        session = None
        try:
            session = _Session()
            session.add(_EvidenceModel(
                id=eid, event_id=event_id, type=evidence_type, storage_path=storage_path,
                mime_type=mime_type, size_bytes=size_bytes, duration_ms=duration_ms,
                track_id=track_id, face_id=face_id, person_id=person_id,
            ))
            session.commit()
            sys_logger.info(f"[EvidenceStore] Metadata salva: evidence_id={eid} event={event_id}")
            return eid
        except Exception as e:
            sys_logger.error(f"[EvidenceStore] Falha ao salvar no banco: {e}")
        finally:
            if session:
                session.close()

    _save_fallback(evidence_id=eid, event_id=event_id, storage_path=storage_path,
                   mime_type=mime_type, size_bytes=size_bytes, duration_ms=duration_ms,
                   evidence_type=evidence_type, track_id=track_id, face_id=face_id, person_id=person_id)
    return eid


def _save_fallback(**kwargs):
    os.makedirs(EVIDENCES_DIR, exist_ok=True)
    try:
        with open(FALLBACK_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **kwargs},
                               ensure_ascii=False, default=str) + "\n")
        sys_logger.info(f"[EvidenceStore] Metadata salva em fallback: event_id={kwargs.get('event_id')}")
    except Exception as e:
        sys_logger.error(f"[EvidenceStore] Falha no fallback JSONL: {e}")
