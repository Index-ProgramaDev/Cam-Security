import time
import winsound
from loguru import logger

class AlertManager:
    """
    Gerenciador de alertas com controle de spam (debouncing).
    Evita gravar múltiplos alertas repetidos seguidos para o mesmo indivíduo e evento.
    """
    def __init__(self, event_logger, cooldown_seconds=15.0):
        self.event_logger = event_logger
        self.cooldown_seconds = cooldown_seconds
        self.active_alerts = {}  # (track_id, event_type) -> timestamp do último disparo
        
        # Buffer de alertas na memória para atualização em tempo real do dashboard
        self.live_alerts = []
        self.max_live_alerts = 50

    def trigger_alert(self, event_type, track_id, risk_score, evidence_path=None, description=None):
        """
        Dispara um alerta se o cooldown tiver expirado.
        Grava no banco de dados e adiciona ao buffer de tempo real.
        """
        now = time.time()
        key = (track_id, event_type)
        
        last_triggered = self.active_alerts.get(key, 0.0)
        
        # Se ultrapassou o tempo de cooldown
        if now - last_triggered > self.cooldown_seconds:
            # Registra no banco de dados
            self.event_logger.log_event(
                event_type=event_type,
                track_id=track_id,
                risk_score=risk_score,
                evidence_path=evidence_path,
                description=description
            )
            
            # Atualiza timestamp
            self.active_alerts[key] = now
            
            # Adiciona ao buffer para o frontend em tempo real
            alert_entry = {
                "timestamp": time.strftime("%H:%M:%S"),
                "event_type": event_type,
                "track_id": track_id,
                "risk_score": risk_score,
                "evidence_path": evidence_path if evidence_path else "",
                "description": description if description else ""
            }
            
            self.live_alerts.append(alert_entry)
            if len(self.live_alerts) > self.max_live_alerts:
                self.live_alerts.pop(0)
                
            return True
            
        return False

    def clean_stale_alerts(self, active_track_ids):
        """Remove registros de cooldown para IDs que já saíram do monitoramento."""
        stale_keys = [key for key in self.active_alerts.keys() if key[0] not in active_track_ids]
        for key in stale_keys:
            self.active_alerts.pop(key, None)

    def get_live_alerts(self):
        """Retorna os alertas na memória."""
        return list(self.live_alerts)

    def play_alert_sound(self):
        """Toca um alerta sonoro usando winsound (Windows)."""
        try:
            # Tocar beep (frequência 1000 Hz, duração 500 ms)
            winsound.Beep(1000, 500)
        except Exception as e:
            logger.error(f"Erro ao tocar alerta sonoro: {e}")
