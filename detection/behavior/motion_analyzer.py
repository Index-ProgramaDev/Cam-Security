import math
from typing import List
from .pose_history import TrackSnapshot

class MotionAnalyzer:
    """Analisa histórico temporal para extrair velocidade, direção e aceleração de forma normalizada."""
    
    @staticmethod
    def calculate_distance(p1, p2) -> float:
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

    @classmethod
    def update_kinematics(cls, history: List[TrackSnapshot]):
        """Atualiza a velocidade e aceleração do último snapshot com base nos anteriores.
        A velocidade é normalizada dividindo a distância pela altura da bounding box.
        """
        if len(history) < 2:
            return

        curr = history[-1]
        prev = history[-2]
        dt = curr.timestamp - prev.timestamp
        
        if dt <= 0.001:
            curr.velocity = prev.velocity
            curr.acceleration = prev.acceleration
            return
            
        # Velocidade bruta em pixels/segundo
        vx_px = (curr.center[0] - prev.center[0]) / dt
        vy_px = (curr.center[1] - prev.center[1]) / dt
        
        # Normalização usando a altura (que costuma ser mais estável que largura para humanos)
        norm_factor = curr.height
        
        vx_norm = vx_px / norm_factor
        vy_norm = vy_px / norm_factor
        
        curr.velocity = (vx_norm, vy_norm)
        
        # Calcula aceleração se tivermos pelo menos 3 frames
        if len(history) >= 3:
            ax_norm = (curr.velocity[0] - prev.velocity[0]) / dt
            ay_norm = (curr.velocity[1] - prev.velocity[1]) / dt
            curr.acceleration = (ax_norm, ay_norm)
        else:
            curr.acceleration = (0.0, 0.0)

    @classmethod
    def get_speed(cls, velocity: tuple) -> float:
        """Retorna a magnitude do vetor velocidade."""
        return math.sqrt(velocity[0]**2 + velocity[1]**2)
        
    @classmethod
    def get_average_speed(cls, history: List[TrackSnapshot], window: float = 1.0) -> float:
        """Calcula a velocidade média no último 'window' segundos."""
        if not history: return 0.0
        
        curr_time = history[-1].timestamp
        speeds = [
            cls.get_speed(s.velocity) for s in history 
            if (curr_time - s.timestamp) <= window
        ]
        
        if not speeds: return 0.0
        return sum(speeds) / len(speeds)
