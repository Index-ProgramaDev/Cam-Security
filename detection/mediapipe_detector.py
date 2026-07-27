import cv2
import time
import os
import mediapipe as mp
from utils.logger import sys_logger

class MediaPipeDetector:
    def __init__(self):
        self.detector = None
        self.ready = False
        self.last_timestamp_ms = 0
        self._init_detector()

    def _init_detector(self):
        try:
            model_path = "pose_landmarker_lite.task"
            if not os.path.exists(model_path):
                model_path = "pose_landmarker_full.task"

            if os.path.exists(model_path):
                base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
                options = mp.tasks.vision.PoseLandmarkerOptions(
                    base_options=base_options,
                    running_mode=mp.tasks.vision.RunningMode.VIDEO,
                    num_poses=3,
                    min_pose_detection_confidence=0.5,
                    min_pose_presence_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)
                self.ready = True
                sys_logger.info(f"MediaPipe PoseLandmarker otimizado inicializado com {model_path}.")
            else:
                sys_logger.warning("Modelo pose_landmarker_lite.task não encontrado. pose detection em standby.")
        except Exception as e:
            sys_logger.error(f"Erro ao carregar MediaPipe PoseDetector: {e}")
            self.ready = False

    def process(self, frame):
        if not self.ready or frame is None:
            return {"pose_landmarks": []}

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            now_ms = int(time.time() * 1000)
            if now_ms <= self.last_timestamp_ms:
                now_ms = self.last_timestamp_ms + 1
            self.last_timestamp_ms = now_ms

            res = self.detector.detect_for_video(mp_img, now_ms)
            landmarks_list = res.pose_landmarks if res.pose_landmarks else []
            return {"pose_landmarks": landmarks_list}

        except Exception as e:
            sys_logger.error(f"Erro ao processar pose: {e}")
            return {"pose_landmarks": []}

    def close(self):
        if self.ready and self.detector:
            self.detector.close()
            self.ready = False
