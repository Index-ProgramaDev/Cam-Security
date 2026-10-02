import time
from typing import Dict, List, Any

from utils.logger import sys_logger

from .pose_history import PoseHistory, TrackSnapshot
from .motion_analyzer import MotionAnalyzer
from .speed_detector import SpeedDetector
from .punch_detector import PunchDetector
from .collision_detector import CollisionDetector
from .fight_detector import FightDetector
from .pursuit_detector import PursuitDetector

from .config import (
    PUNCH_COOLDOWN, FIGHT_COOLDOWN, PURSUIT_COOLDOWN, COLLISION_COOLDOWN,
    MIN_STABLE_HISTORY,
)


class BehaviorEngine:
    """Orquestra a detecção comportamental.

    Fase 1 — Sinais individuais (nunca vão ao AlertManager):
        high_speed, sudden_movement

    Fase 2 — Eventos individuais (vão ao AlertManager):
        SOCO  (depende de sudden_movement + wrist signals)

    Fase 3 — Eventos compostos por pares (vão ao AlertManager):
        LUTA, PERSEGUICAO, COLISAO
    """

    def __init__(self):
        self.history   = PoseHistory()
        self.cooldowns: Dict[str, float] = {}

    def _on_cd(self, key: str, duration: float, now: float) -> bool:
        return (now - self.cooldowns.get(key, 0.0)) < duration

    def _set_cd(self, key: str, now: float):
        self.cooldowns[key] = now

    def process_frame(self, tracks: Dict[int, Any], timestamp: float) -> List[Dict[str, Any]]:
        self.history.cleanup(timestamp)
        events: List[Dict[str, Any]] = []
        signals: Dict[int, Dict] = {}

        # ---- Atualiza histórico e cinemática ----
        active_ids = list(tracks.keys())
        for tid, info in tracks.items():
            snap = TrackSnapshot(
                timestamp=timestamp,
                box=info["box"],
                pose=info.get("pose"),
                confidence=info.get("face_confidence", 1.0),
            )
            self.history.add_snapshot(tid, snap)
            hist = self.history.get_track_history(tid)
            MotionAnalyzer.update_kinematics(hist)

            signals[tid] = {"high_speed": False, "sudden_movement": False, "punch": False}

        total_tracks = len(active_ids)

        # ---- FASE 1 + 2: Sinais e eventos individuais ----
        for tid in active_ids:
            hist = self.history.get_track_history(tid)
            if len(hist) < MIN_STABLE_HISTORY:
                continue

            high_spd, sudden_mov, _ = SpeedDetector.evaluate(hist)
            signals[tid]["high_speed"]      = high_spd
            signals[tid]["sudden_movement"] = sudden_mov

            is_punch, punch_conf = PunchDetector.evaluate(hist, signals[tid])
            if is_punch:
                signals[tid]["punch"] = True
                if not self._on_cd(f"PUNCH_{tid}", PUNCH_COOLDOWN, timestamp):
                    events.append({
                        "type": "SOCO", "track_id": tid,
                        "confidence": punch_conf,
                    })
                    self._set_cd(f"PUNCH_{tid}", timestamp)

        # ---- FASE 3: Eventos compostos (pares) ----
        for i in range(total_tracks):
            for j in range(i + 1, total_tracks):
                idA, idB   = active_ids[i], active_ids[j]
                histA      = self.history.get_track_history(idA)
                histB      = self.history.get_track_history(idB)
                sigA, sigB = signals[idA], signals[idB]
                pair_key   = f"{min(idA,idB)}_{max(idA,idB)}"

                # Colisão
                is_col, col_conf = CollisionDetector.evaluate(histA, histB)
                if is_col and not self._on_cd(f"COL_{pair_key}", COLLISION_COOLDOWN, timestamp):
                    events.append({
                        "type": "COLISAO", "track_id": idA,
                        "related_ids": [idB], "confidence": col_conf,
                    })
                    self._set_cd(f"COL_{pair_key}", timestamp)

                # Luta
                is_fight, fight_conf = FightDetector.evaluate(histA, histB, sigA, sigB)
                if is_fight and not self._on_cd(f"FIGHT_{pair_key}", FIGHT_COOLDOWN, timestamp):
                    events.append({
                        "type": "LUTA", "track_id": idA,
                        "related_ids": [idB], "confidence": fight_conf,
                    })
                    self._set_cd(f"FIGHT_{pair_key}", timestamp)

                # Perseguição (com total_tracks para o gate)
                is_pursuit, pursuit_conf = PursuitDetector.evaluate(histA, histB, total_tracks)
                if is_pursuit and not self._on_cd(f"PURSUIT_{pair_key}", PURSUIT_COOLDOWN, timestamp):
                    events.append({
                        "type": "PERSEGUICAO", "track_id": idA,
                        "related_ids": [idB], "confidence": pursuit_conf,
                    })
                    self._set_cd(f"PURSUIT_{pair_key}", timestamp)

        return events
