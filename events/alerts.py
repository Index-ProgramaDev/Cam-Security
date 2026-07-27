import time
import winsound
from utils.logger import sys_logger

class AlertManager:
    """
    Gerenciador de alertas refatorado com controle de debouncing (cooldown) e despacho integrado.
    """
    def __init__(self, event_logger, notification_dispatcher=None, cooldown_seconds=5.0):
        self.event_logger = event_logger
        self.notification_dispatcher = notification_dispatcher
        self.cooldown_seconds = cooldown_seconds
        self.last_alerts = {}  # (track_id, event_type) -> timestamp

    def trigger_alert(self, event_type: str, track_id: int, risk_score: int = 100, description: str = ""):
        now = time.time()
        key = (track_id, event_type)
        last_time = self.last_alerts.get(key, 0.0)

        if now - last_time < self.cooldown_seconds:
            return False

        self.last_alerts[key] = now

        # Grava log de auditoria
        if self.event_logger:
            self.event_logger.log_event(
                event_type=event_type,
                track_id=track_id,
                risk_score=risk_score,
                description=description
            )

        # Despacha notificação para consumo
        if self.notification_dispatcher:
            self.notification_dispatcher.dispatch(
                event_type=event_type,
                track_id=track_id,
                risk_score=risk_score,
                description=description
            )

        # Toca alerta sonoro
        self.play_alert_sound()
        return True

    def play_alert_sound(self):
        try:
            winsound.Beep(1000, 400)
        except Exception as e:
            sys_logger.error(f"Erro ao tocar beep de alerta: {e}")
