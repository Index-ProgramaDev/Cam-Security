import json
import time
import os
from utils.logger import sys_logger


class NotificationDispatcher:
    """Serializa alertas em JSONL para consumo por interface futura (WebSocket, etc.)."""

    def __init__(self, output_file: str = "events/audit_events.jsonl"):
        self.output_file = output_file
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

    def dispatch(self, event_type: str, track_id: int, risk_score: int, description: str = ""):
        payload = {
            "timestamp":   time.strftime("%Y-%m-%d %H:%M:%S"),
            "event_type":  event_type,
            "track_id":    track_id,
            "risk_score":  risk_score,
            "description": description or "",
        }
        try:
            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            sys_logger.info(f"[NOTIFICAÇÃO] Alerta despachado: Track #{track_id}")
        except Exception as e:
            sys_logger.error(f"Erro ao despachar notificação: {e}")
