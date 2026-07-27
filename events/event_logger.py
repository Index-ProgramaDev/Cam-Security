import time
import json
import os
from utils.logger import sys_logger

class EventLogger:
    """
    Guarda logs de auditoria de segurança em arquivo estruturado.
    """
    def __init__(self, audit_file="events/audit_events.log"):
        self.audit_file = audit_file
        os.makedirs(os.path.dirname(self.audit_file), exist_ok=True)

    def log_event(self, event_type: str, track_id: int, risk_score: int, description: str = "", evidence_path: str = ""):
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "event_type": event_type,
            "track_id": track_id,
            "risk_score": risk_score,
            "description": description or "",
            "evidence_path": evidence_path or ""
        }

        try:
            with open(self.audit_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            sys_logger.info(f"[AUDITORIA] Evento '{event_type}' registrado para Track #{track_id} (Risco: {risk_score})")
        except Exception as e:
            sys_logger.error(f"Erro ao gravar log de auditoria: {e}")
