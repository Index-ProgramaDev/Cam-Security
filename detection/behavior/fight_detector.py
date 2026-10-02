from collections import defaultdict
from typing import List, Tuple, Dict

from utils.logger import sys_logger
from .pose_history import TrackSnapshot
from .motion_analyzer import MotionAnalyzer
from .config import FIGHT_MIN_CONFIDENCE, FIGHT_MAX_DISTANCE, FIGHT_WINDOW, FIGHT_MIN_FRAMES

# ---- Ligar/desligar debug ----
FIGHT_DEBUG = True


class FightDetector:
    _windows: Dict[str, list] = defaultdict(list)

    @classmethod
    def _pair_key(cls, histA, histB) -> str:
        ia, ib = id(histA), id(histB)
        return f"{min(ia,ib)}_{max(ia,ib)}"

    @classmethod
    def evaluate(
        cls,
        histA: List[TrackSnapshot],
        histB: List[TrackSnapshot],
        signalsA: Dict,
        signalsB: Dict,
    ) -> Tuple[bool, float]:
        if not histA or not histB:
            return False, 0.0

        currA = histA[-1]
        currB = histB[-1]

        dist      = MotionAnalyzer.calculate_distance(currA.center, currB.center)
        avg_h     = max(1.0, (currA.height + currB.height) / 2.0)
        dist_norm = dist / avg_h
        speedA    = MotionAnalyzer.get_speed(currA.velocity)
        speedB    = MotionAnalyzer.get_speed(currB.velocity)

        has_punch  = signalsA.get("punch", False) or signalsB.get("punch", False)
        has_sudden = signalsA.get("sudden_movement", False) or signalsB.get("sudden_movement", False)
        has_aggression = has_punch or has_sudden
        has_motion = speedA > 0.8 or speedB > 0.8

        pk = cls._pair_key(histA, histB)
        ts_now = currA.timestamp
        cls._windows[pk].append(ts_now)
        cls._windows[pk] = [t for t in cls._windows[pk] if (ts_now - t) <= FIGHT_WINDOW]
        interaction_count = len(cls._windows[pk])

        if FIGHT_DEBUG:
            sys_logger.debug(
                f"[FIGHT-DBG] dist_norm={dist_norm:.3f}(max={FIGHT_MAX_DISTANCE}) "
                f"speedA={speedA:.3f} speedB={speedB:.3f} "
                f"punch={has_punch} sudden={has_sudden} "
                f"frames={interaction_count}(min={FIGHT_MIN_FRAMES})"
            )

        # Condição 1 — Proximidade
        if dist_norm > FIGHT_MAX_DISTANCE:
            pk and cls._windows.pop(pk, None)
            return False, 0.0

        # Condição 2 — Agressão
        if not has_aggression:
            return False, 0.0

        # Condição 3 — Movimento
        if not has_motion:
            return False, 0.0

        # Condição 4 — Persistência
        if interaction_count < FIGHT_MIN_FRAMES:
            return False, 0.0

        prox_score       = max(0.0, 1.0 - dist_norm / FIGHT_MAX_DISTANCE)
        aggression_score = 0.95 if has_punch else 0.65
        persist_score    = min(1.0, interaction_count / 10.0)
        motion_score     = min(1.0, max(speedA, speedB) / 2.0)

        final_score = (
            prox_score       * 0.25
            + aggression_score * 0.40
            + persist_score    * 0.20
            + motion_score     * 0.15
        )

        if FIGHT_DEBUG:
            sys_logger.debug(
                f"[FIGHT-DBG] PASS score={final_score:.3f}(min={FIGHT_MIN_CONFIDENCE})"
            )

        if final_score < FIGHT_MIN_CONFIDENCE:
            return False, 0.0

        return True, final_score
