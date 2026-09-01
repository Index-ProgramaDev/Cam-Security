import time
import threading
import collections
from typing import Optional, Callable, List, Tuple

import numpy as np

from utils.logger import sys_logger

PRE_TRIGGER_SECONDS  = 5.0
POST_TRIGGER_SECONDS = 15.0
_MAX_DEQUE_SIZE      = int(PRE_TRIGGER_SECONDS * 30 * 1.5)


class TriggerCapture:
    __slots__ = ("event_id", "camera_id", "triggered_at", "deadline", "post_frames", "done", "lock")

    def __init__(self, event_id: str, camera_id: str, triggered_at: float):
        self.event_id     = event_id
        self.camera_id    = camera_id
        self.triggered_at = triggered_at
        self.deadline     = triggered_at + POST_TRIGGER_SECONDS
        self.post_frames: List[Tuple[float, np.ndarray]] = []
        self.done         = False
        self.lock         = threading.Lock()


class VideoBuffer:
    def __init__(self, camera_id: str = "cam_0", fps: float = 30):
        self.camera_id      = camera_id
        self.fps            = fps
        self._frames        = collections.deque(maxlen=_MAX_DEQUE_SIZE)
        self._lock          = threading.Lock()
        self._active:       Optional[TriggerCapture] = None
        self._capture_lock  = threading.Lock()

    def push(self, frame: np.ndarray, timestamp: Optional[float] = None):
        if frame is None:
            return
        ts = timestamp or time.time()
        with self._lock:
            self._frames.append((ts, frame))
        with self._capture_lock:
            cap = self._active
        if cap and not cap.done:
            with cap.lock:
                if not cap.done:
                    cap.post_frames.append((ts, frame))
                    if time.time() >= cap.deadline:
                        cap.done = True

    def trigger(self, event_id: str, on_ready_callback: Optional[Callable] = None) -> Optional[TriggerCapture]:
        with self._capture_lock:
            if self._active and not self._active.done:
                sys_logger.warning(f"[VideoBuffer:{self.camera_id}] Captura já ativa para event {self._active.event_id} — ignorando {event_id}")
                return None

        triggered_at = time.time()
        cap = TriggerCapture(event_id, self.camera_id, triggered_at)

        with self._lock:
            pre_frames = [(ts, f) for ts, f in self._frames if ts >= triggered_at - PRE_TRIGGER_SECONDS]

        sys_logger.info(f"[VideoBuffer:{self.camera_id}] Gatilho! event={event_id} pre_frames={len(pre_frames)} coletando +{POST_TRIGGER_SECONDS}s...")

        with self._capture_lock:
            self._active = cap

        if on_ready_callback:
            def _run():
                deadline = triggered_at + POST_TRIGGER_SECONDS + 2.0
                while time.time() < deadline:
                    with cap.lock:
                        if cap.done:
                            break
                    time.sleep(0.1)
                with cap.lock:
                    cap.done = True
                with self._capture_lock:
                    if self._active is cap:
                        self._active = None
                try:
                    on_ready_callback(pre_frames, cap.post_frames, event_id, triggered_at)
                except Exception as e:
                    sys_logger.error(f"[VideoBuffer:{self.camera_id}] Erro no callback: {e}")

            threading.Thread(target=_run, daemon=True).start()

        return cap

    def is_capturing(self) -> bool:
        with self._capture_lock:
            return bool(self._active and not self._active.done)

    def get_buffer_duration(self) -> float:
        with self._lock:
            return (self._frames[-1][0] - self._frames[0][0]) if len(self._frames) >= 2 else 0.0

    def clear(self):
        with self._lock:
            self._frames.clear()
