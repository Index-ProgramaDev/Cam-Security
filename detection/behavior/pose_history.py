import time
from collections import deque
from typing import Dict, List, Any, Optional

from utils.logger import sys_logger
from .config import HISTORY_WINDOW_SECONDS, HISTORY_CLEANUP_INTERVAL

class TrackSnapshot:
    """Representa o estado de um track em um frame específico."""
    __slots__ = ("timestamp", "box", "center", "width", "height", "pose", "confidence", "velocity", "acceleration")
    
    def __init__(self, timestamp: float, box: List[float], pose: Any, confidence: float = 1.0):
        self.timestamp = timestamp
        self.box = box # [x1, y1, x2, y2]
        self.center = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
        self.width = max(1.0, box[2] - box[0])
        self.height = max(1.0, box[3] - box[1])
        self.pose = pose
        self.confidence = confidence
        
        # Estes campos são calculados posteriormente pelo MotionAnalyzer
        self.velocity = (0.0, 0.0)      # (vx, vy) normalizado ou pixels/s
        self.acceleration = (0.0, 0.0)  # (ax, ay) normalizado ou pixels/s^2

class PoseHistory:
    """Mantém o histórico temporal limitado (sliding window) por track_id."""
    def __init__(self):
        self.history: Dict[int, deque] = {}
        self._last_cleanup = time.time()

    def add_snapshot(self, track_id: int, snapshot: TrackSnapshot):
        if track_id not in self.history:
            self.history[track_id] = deque()
            
        self.history[track_id].append(snapshot)
        self._prune_track(track_id, snapshot.timestamp)
        
    def _prune_track(self, track_id: int, current_time: float):
        """Remove snapshots mais antigos que a HISTORY_WINDOW_SECONDS."""
        dq = self.history.get(track_id)
        if not dq: return
        
        while dq and (current_time - dq[0].timestamp) > HISTORY_WINDOW_SECONDS:
            dq.popleft()

    def cleanup(self, current_time: Optional[float] = None):
        """Limpeza geral de tracks inativos."""
        now = current_time or time.time()
        if (now - self._last_cleanup) < HISTORY_CLEANUP_INTERVAL:
            return
            
        inactive_ids = []
        for tid, dq in self.history.items():
            if not dq or (now - dq[-1].timestamp) > HISTORY_WINDOW_SECONDS:
                inactive_ids.append(tid)
                
        for tid in inactive_ids:
            del self.history[tid]
            
        self._last_cleanup = now

    def get_track_history(self, track_id: int) -> List[TrackSnapshot]:
        """Retorna uma cópia lista do histórico atual do track."""
        return list(self.history.get(track_id, []))
