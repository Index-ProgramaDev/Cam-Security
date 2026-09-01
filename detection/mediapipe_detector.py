import cv2
import time
import os
import mediapipe as mp
from utils.logger import sys_logger

MIN_CROP_SIZE           = 80
POSE_HOLD_TTL           = 0.35
MIN_LANDMARK_VISIBILITY = 0.30
_DEBUG_INTERVAL         = 90


class Landmark:
    __slots__ = ("x", "y", "z", "visibility", "presence")

    def __init__(self, x, y, z=0.0, visibility=1.0, presence=1.0):
        self.x          = float(x)
        self.y          = float(y)
        self.z          = float(z)
        self.visibility = float(visibility)
        self.presence   = float(presence)


def _copy_landmarks(raw):
    return [
        Landmark(lm.x, lm.y, getattr(lm, "z", 0.0),
                 getattr(lm, "visibility", 1.0), getattr(lm, "presence", 1.0))
        for lm in raw
    ]


def map_crop_landmarks_to_frame(landmarks, crop_box, frame_w, frame_h):
    if not landmarks:
        return []
    x1, y1, x2, y2 = crop_box
    cw, ch = max(1, x2 - x1), max(1, y2 - y1)
    fw, fh = max(frame_w, 1), max(frame_h, 1)
    return [
        Landmark((x1 + lm.x * cw) / fw, (y1 + lm.y * ch) / fh,
                 lm.z, lm.visibility, lm.presence)
        for lm in landmarks
    ]


def _build_options(model_path):
    return mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.45,
        min_pose_presence_confidence=0.45,
        min_tracking_confidence=0.45,
        output_segmentation_masks=False,
    )


class PoseHold:
    def __init__(self, ttl_seconds: float = POSE_HOLD_TTL):
        self.ttl_seconds = ttl_seconds
        self._last: dict = {}

    def update(self, track_id: int, landmarks: list):
        now = time.time()
        if landmarks:
            self._last[track_id] = (landmarks, now)
            return landmarks
        prev = self._last.get(track_id)
        return prev[0] if prev and (now - prev[1]) <= self.ttl_seconds else None

    def prune(self, active_ids: set):
        for tid in [t for t in self._last if t not in active_ids]:
            del self._last[tid]


class _TrackLandmarker:
    def __init__(self, model_path: str):
        self.landmarker    = mp.tasks.vision.PoseLandmarker.create_from_options(_build_options(model_path))
        self._last_ts_ms:  int = 0
        self._frame_count: int = 0

    def detect(self, rgb_crop, frame_ts_ms: int) -> list:
        ts_ms = max(frame_ts_ms, self._last_ts_ms + 1)
        self._last_ts_ms   = ts_ms
        self._frame_count += 1

        result = self.landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop), ts_ms
        )
        if not result.pose_landmarks:
            return []

        landmarks = _copy_landmarks(result.pose_landmarks[0])

        if _DEBUG_INTERVAL > 0 and (self._frame_count % _DEBUG_INTERVAL) == 1:
            n = len(landmarks)
            pts = {i: landmarks[i] for i in (0, 11, 12, 23) if i < n}
            extra = " ".join(f"lm{i}=({p.x:.3f},{p.y:.3f})" for i, p in pts.items())
            sys_logger.debug(f"[MediaPipe|diag] ts={ts_ms}ms landmarks={n} {extra}")

        return landmarks

    def close(self):
        try:
            self.landmarker.close()
        except Exception:
            pass


class MediaPipeDetector:
    def __init__(self):
        self._model_path = ""
        self._pool: dict = {}
        self.ready       = False
        self.pose_hold   = PoseHold()
        self._init_model()

    def _init_model(self):
        for candidate in ("pose_landmarker_lite.task", "pose_landmarker_full.task"):
            if os.path.exists(candidate):
                self._model_path = candidate
                self.ready       = True
                sys_logger.info(f"[MediaPipe] Modelo encontrado: {candidate} (RunningMode.VIDEO, 1 pose/crop)")
                return
        sys_logger.warning("[MediaPipe] Nenhum modelo .task encontrado. Pose em standby.")

    def _get_landmarker(self, track_id: int):
        if not self.ready:
            return None
        if track_id not in self._pool:
            try:
                self._pool[track_id] = _TrackLandmarker(self._model_path)
                sys_logger.debug(f"[MediaPipe] Detector criado para track #{track_id}")
            except Exception as e:
                sys_logger.error(f"[MediaPipe] Falha ao criar detector #{track_id}: {e}")
                return None
        return self._pool[track_id]

    def process_crop(self, track_id: int, crop, frame_ts_ms: int) -> list:
        if crop is None or crop.size == 0:
            return []
        h, w = crop.shape[:2]
        if w < MIN_CROP_SIZE or h < MIN_CROP_SIZE:
            return []
        ld = self._get_landmarker(track_id)
        if ld is None:
            return []
        try:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            if not rgb.flags["C_CONTIGUOUS"]:
                rgb = rgb.copy()
            return ld.detect(rgb, frame_ts_ms)
        except Exception as e:
            sys_logger.error(f"[MediaPipe] Erro no crop track #{track_id}: {e}")
            return []

    def process_for_track(self, track_id: int, crop, crop_box,
                          frame_w: int, frame_h: int, frame_ts_ms: int = 0):
        crop_lms = self.process_crop(track_id, crop, frame_ts_ms)
        mapped   = map_crop_landmarks_to_frame(crop_lms, crop_box, frame_w, frame_h) if crop_lms else []
        return self.pose_hold.update(track_id, mapped)

    def prune_pool(self, active_ids: set):
        for tid in [t for t in self._pool if t not in active_ids]:
            try:
                self._pool[tid].close()
            except Exception:
                pass
            del self._pool[tid]
            sys_logger.debug(f"[MediaPipe] Detector removido para track #{tid}")

    def close(self):
        for ld in self._pool.values():
            try:
                ld.close()
            except Exception:
                pass
        self._pool.clear()
        self.ready = False
