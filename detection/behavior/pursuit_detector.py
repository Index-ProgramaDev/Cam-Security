from typing import List, Tuple

from .pose_history import TrackSnapshot
from .motion_analyzer import MotionAnalyzer
from .config import (
    PURSUIT_MIN_CONFIDENCE, PURSUIT_MIN_SPEED,
    PURSUIT_MIN_DISTANCE, PURSUIT_MAX_DISTANCE,
    PURSUIT_SIMILAR_SPEED, PURSUIT_WINDOW,
)


class PursuitDetector:
    """Detecta PERSEGUIÇÃO entre dois tracks.

    Condições TODAS obrigatórias:
        1. Total de tracks na cena <= PURSUIT_MAX_TRACKS  (passado pelo engine)
        2. Distância entre A e B dentro de [MIN_DISTANCE, MAX_DISTANCE] normalizados
        3. Ambos com velocidade >= PURSUIT_MIN_SPEED
        4. Mesma direção (cos_theta > 0.80)
        5. Distância DIMINUINDO ao longo da janela (não estável sozinha)
        6. NÃO são "amigos": velocidades muito semelhantes + dist muito pequena é excluído
        7. Relação perseguidor/alvo: A está atrás de B e A é mais rápido que B
    """

    @staticmethod
    def _norm_dist(snapA: TrackSnapshot, snapB: TrackSnapshot) -> float:
        d = MotionAnalyzer.calculate_distance(snapA.center, snapB.center)
        avg_h = max(1.0, (snapA.height + snapB.height) / 2.0)
        return d / avg_h

    @classmethod
    def evaluate(
        cls,
        histA: List[TrackSnapshot],
        histB: List[TrackSnapshot],
        total_tracks: int,
    ) -> Tuple[bool, float]:
        from .config import PURSUIT_MAX_TRACKS

        # Condição 1 — Limite de tracks na cena
        if total_tracks > PURSUIT_MAX_TRACKS:
            return False, 0.0

        if len(histA) < 4 or len(histB) < 4:
            return False, 0.0

        currA, currB = histA[-1], histB[-1]
        curr_dist_norm = cls._norm_dist(currA, currB)

        # Condição 2 — Distância válida (nem muito perto nem muito longe)
        if curr_dist_norm < PURSUIT_MIN_DISTANCE or curr_dist_norm > PURSUIT_MAX_DISTANCE:
            return False, 0.0

        speedA = MotionAnalyzer.get_speed(currA.velocity)
        speedB = MotionAnalyzer.get_speed(currB.velocity)

        # Condição 3 — Ambos em movimento
        if speedA < PURSUIT_MIN_SPEED or speedB < PURSUIT_MIN_SPEED:
            return False, 0.0

        # Condição 6 — Exclusão de "amigos" (velocidades próximas + distância pequena)
        speed_diff = abs(speedA - speedB)
        if speed_diff < PURSUIT_SIMILAR_SPEED and curr_dist_norm < PURSUIT_MIN_DISTANCE * 1.8:
            return False, 0.0

        # Condição 4 — Mesma direção
        dot_prod = (
            currA.velocity[0] * currB.velocity[0]
            + currA.velocity[1] * currB.velocity[1]
        )
        mag_prod = speedA * speedB
        if mag_prod == 0:
            return False, 0.0

        cos_theta = dot_prod / mag_prod
        if cos_theta <= 0.80:
            return False, 0.0

        # Condição 5 — Distância DIMINUINDO ao longo da janela temporal
        curr_time = currA.timestamp
        wA = [s for s in histA if (curr_time - s.timestamp) <= PURSUIT_WINDOW]
        wB = [s for s in histB if (curr_time - s.timestamp) <= PURSUIT_WINDOW]

        if len(wA) < 3 or len(wB) < 3:
            return False, 0.0

        oldest_dist = cls._norm_dist(wA[0], wB[0])
        newest_dist = cls._norm_dist(wA[-1], wB[-1])

        # Distância precisa diminuir (pelo menos 0.15 unidades normalizadas)
        if oldest_dist - newest_dist < 0.15:
            return False, 0.0

        # Condição 7 — Relação perseguidor/alvo (A mais rápido e "atrás" de B)
        # Determinamos quem está "atrás" pelo vetor de A→B vs direção do movimento
        dx_ab = currB.center[0] - currA.center[0]
        dy_ab = currB.center[1] - currA.center[1]
        dist_ab = MotionAnalyzer.calculate_distance(currA.center, currB.center)
        if dist_ab == 0:
            return False, 0.0

        # Produto escalar entre movimento de A e vetor A→B
        # Se positivo: A está se movendo EM DIREÇÃO a B (A atrás de B)
        dot_a_toward_b = (
            currA.velocity[0] * (dx_ab / dist_ab)
            + currA.velocity[1] * (dy_ab / dist_ab)
        )

        is_a_chasing = dot_a_toward_b > 0.5  # A indo em direção a B
        is_a_faster  = speedA > speedB * 1.05  # A pelo menos 5% mais rápido

        if not (is_a_chasing or is_a_faster):
            return False, 0.0

        # Score composto
        direction_score  = min(1.0, (cos_theta - 0.80) / 0.20)
        approach_score   = min(1.0, (oldest_dist - newest_dist) / max(0.1, oldest_dist))
        speed_score      = min(1.0, min(speedA, speedB) / (PURSUIT_MIN_SPEED * 2.0))
        chaser_bonus     = 0.15 if (is_a_chasing and is_a_faster) else 0.0

        final_score = (
            direction_score * 0.35
            + approach_score  * 0.35
            + speed_score     * 0.20
            + chaser_bonus    * 0.10
        )

        if final_score < PURSUIT_MIN_CONFIDENCE:
            return False, 0.0

        return True, final_score
