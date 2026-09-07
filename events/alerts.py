"""
AlertManager — dispara alertas com cooldown e aciona captura de evidência.

Novidades em relação à versão anterior:
  - triggered_at registrado com timestamp Unix preciso do frame/gatilho.
  - evidence_manager opcional: quando fornecido, cada alerta aciona captura
    de vídeo (buffer -5s + trigger + +15s) via EvidenceManager.on_event().
  - face_id e person_id propagados do estado do track para o evento.
  - winsound.Beep encapsulado com fallback silencioso em não-Windows.
"""

import time
import uuid
from typing import Optional

from utils.logger import sys_logger


class AlertManager:
    """
    Gerenciador de alertas com debouncing (cooldown) e integração com evidências.

    Parameters
    ----------
    event_logger           : EventLogger — grava auditoria em arquivo
    notification_dispatcher: NotificationDispatcher — serializa para JSON
    cooldown_seconds       : float — tempo mínimo entre alertas do mesmo tipo/track
    evidence_manager       : EvidenceManager | None — se fornecido, captura vídeo
    camera_id              : str — ID da câmera usada na captura de evidência
    face_storage           : FaceStorage | None — para recuperar face_id/person_id
    face_snapshot          : FaceSnapshotBuffer | None — promove buffer facial no gatilho
    """

    def __init__(
        self,
        event_logger,
        notification_dispatcher=None,
        cooldown_seconds: float = 5.0,
        evidence_manager=None,
        camera_id: str = "cam_0",
        face_storage=None,
        face_snapshot=None,
    ):
        self.event_logger = event_logger
        self.notification_dispatcher = notification_dispatcher
        self.cooldown_seconds = cooldown_seconds
        self.evidence_manager = evidence_manager
        self.camera_id = camera_id
        self.face_storage = face_storage
        self.face_snapshot = face_snapshot

        # (track_id, event_type) -> último timestamp de disparo
        self.last_alerts: dict = {}

    def trigger_alert(
        self,
        event_type: str,
        track_id: int,
        risk_score: int = 100,
        description: str = "",
        triggered_at: Optional[float] = None,
        face_id: Optional[str] = None,
        person_id: Optional[str] = None,
    ) -> bool:
        """
        Dispara um alerta de segurança.

        Parameters
        ----------
        event_type   : tipo do evento ("SOCO", "COLISAO", "FALLEN", etc.)
        track_id     : ID do track envolvido
        risk_score   : score de risco 0-100
        description  : descrição legível
        triggered_at : timestamp Unix do frame onde o evento foi detectado.
                       Se None, usa time.time() no momento da chamada.
        face_id      : face_id associado (preenchido automaticamente se face_storage disponível)
        person_id    : person_id associado (preenchido automaticamente se face_storage disponível)

        Returns
        -------
        True se o alerta foi disparado, False se estava em cooldown.
        """
        now = time.time()
        key = (track_id, event_type)
        last_time = self.last_alerts.get(key, 0.0)

        if now - last_time < self.cooldown_seconds:
            return False

        self.last_alerts[key] = now
        ts = triggered_at if triggered_at is not None else now

        # Resolve face_id/person_id via FaceStorage se não fornecidos
        if self.face_storage is not None:
            if face_id is None:
                face_id = self.face_storage.get_face_id_for_track(track_id)
            if person_id is None:
                person_id = self.face_storage.get_identity_for_track(track_id)

        # Gera event_id único para correlação event ↔ evidence
        event_id = str(uuid.uuid4())[:12]

        # Grava log de auditoria
        if self.event_logger:
            self.event_logger.log_event(
                event_type=event_type,
                track_id=track_id,
                risk_score=risk_score,
                description=description,
                evidence_path="",  # preenchido depois quando vídeo ficar pronto
            )

        # Despacha notificação
        if self.notification_dispatcher:
            self.notification_dispatcher.dispatch(
                event_type=event_type,
                track_id=track_id,
                risk_score=risk_score,
                description=description,
            )

        # Aciona captura de evidência em background
        if self.evidence_manager is not None:
            self.evidence_manager.on_event(
                event_id=event_id,
                camera_id=self.camera_id,
                track_id=track_id,
                face_id=face_id,
                person_id=person_id,
            )

        # Promove buffer facial temporário do track (fotos + embeddings)
        if self.face_snapshot is not None:
            self.face_snapshot.on_trigger(track_id=track_id, event_id=event_id)

        # Log completo com contexto
        sys_logger.info(
            f"[ALERTA] {event_type} | track=#{track_id} "
            f"risk={risk_score} face={face_id} person={person_id} "
            f"event_id={event_id}"
        )

        # Sinal sonoro
        self.play_alert_sound()
        return True

    def play_alert_sound(self):
        try:
            import winsound
            winsound.Beep(1000, 400)
        except Exception:
            pass  # Silencia em não-Windows ou quando winsound não está disponível
