import math
from typing import List, Tuple

from .pose_history import TrackSnapshot
from .motion_analyzer import MotionAnalyzer
from .config import (
    COLLISION_MIN_CONFIDENCE, COLLISION_RELATIVE_SPEED, COLLISION_CLOSE_THRESHOLD,
)


def _norm_dist(snapA: TrackSnapshot, snapB: TrackSnapshot) -> float:
    d = MotionAnalyzer.calculate_distance(snapA.center, snapB.center)
    avg_h = max(1.0, (snapA.height + snapB.height) / 2.0)
    return d / avg_h


def _box_iou(boxA, boxB) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = max(1.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return inter / (areaA + areaB - inter)


class CollisionDetector:
    """Detecta COLISÃO entre dois tracks.

    Condições TODAS obrigatórias:
        1. Velocidade relativa >= COLLISION_RELATIVE_SPEED (aproximação mútua)
        2. Trajetórias convergentes (distância diminuiu na janela)
        3. Muito perto agora (dist_norm <= CLOSE_THRESHOLD OU IoU > 0)
        4. Mudança brusca de velocidade após contato (desaceleração de pelo menos um)
    """

    @staticmethod
    def evaluate(histA: List[TrackSnapshot], histB: List[TrackSnapshot]) -> Tuple[bool, float]:
        if len(histA) < 4 or len(histB) < 4:
            return False, 0.0

        currA, currB = histA[-1], histB[-1]
        prevA, prevB = histA[-2], histB[-2]

        # ---- Condição 3 — proximidade atual ----
        curr_dist_norm = _norm_dist(currA, currB)
        iou = _box_iou(currA.box, currB.box)

        is_close = curr_dist_norm <= COLLISION_CLOSE_THRESHOLD or iou > 0.0
        if not is_close:
            return False, 0.0

        # ---- Condição 2 — trajetórias convergentes (distância diminuiu) ----
        prev_dist_norm = _norm_dist(prevA, prevB)
        converging = prev_dist_norm > curr_dist_norm + 0.08
        if not converging:
            return False, 0.0

        # ---- Condição 1 — velocidade relativa alta ----
        vxA, vyA = currA.velocity
        vxB, vyB = currB.velocity
        rel_speed = MotionAnalyzer.get_speed((vxA - vxB, vyA - vyB))

        if rel_speed < COLLISION_RELATIVE_SPEED:
            return False, 0.0

        # ---- Condição 4 — mudança brusca de velocidade pós-contato ----
        # Pelo menos um dos dois deve ter desacelerado (aceleração negativa / mudança de direção)
        accel_a = MotionAnalyzer.get_speed(currA.acceleration)
        accel_b = MotionAnalyzer.get_speed(currB.acceleration)
        has_post_impact = accel_a > 1.5 or accel_b > 1.5

        # Score composto
        speed_score  = min(1.0, rel_speed / (COLLISION_RELATIVE_SPEED * 2.0))
        prox_score   = max(iou, max(0.0, 1.0 - curr_dist_norm / COLLISION_CLOSE_THRESHOLD))
        conv_score   = min(1.0, (prev_dist_norm - curr_dist_norm) / max(0.1, prev_dist_norm))
        impact_bonus = 0.10 if has_post_impact else 0.0

        final_score = (
            speed_score  * 0.40
            + prox_score   * 0.30
            + conv_score   * 0.20
            + impact_bonus * 0.10
        )

        if final_score < COLLISION_MIN_CONFIDENCE:
            return False, 0.0

        return True, final_score
