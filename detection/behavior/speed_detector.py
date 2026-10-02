from typing import List, Tuple
from .pose_history import TrackSnapshot
from .motion_analyzer import MotionAnalyzer
from .config import HIGH_SPEED_THRESHOLD, SUDDEN_MOVEMENT_THRESHOLD

class SpeedDetector:
    @staticmethod
    def evaluate(history: List[TrackSnapshot]) -> Tuple[bool, bool, float]:
        """Avalia velocidade alta e movimento brusco.
        Retorna (is_high_speed, is_sudden_movement, max_score)
        """
        if len(history) < 3:
            return False, False, 0.0
            
        curr = history[-1]
        speed = MotionAnalyzer.get_speed(curr.velocity)
        accel = MotionAnalyzer.get_speed(curr.acceleration)
        
        is_high_speed = speed > HIGH_SPEED_THRESHOLD
        is_sudden_movement = accel > SUDDEN_MOVEMENT_THRESHOLD
        
        score = min(1.0, max(speed / (HIGH_SPEED_THRESHOLD * 1.5), accel / (SUDDEN_MOVEMENT_THRESHOLD * 1.5)))
        
        return is_high_speed, is_sudden_movement, score
