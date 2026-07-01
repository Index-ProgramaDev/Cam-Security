import cv2
import numpy as np
import mediapipe as mp
import time
from loguru import logger


class MediaPipeDetector:
    def __init__(self):
        self._init_tasks_api()
        self.hand_open_frames = 0
        self.hand_closed_frames = 0
        self.is_hand_open_state = False
        self.frames_required_open = 30
        self.alert_triggered = False

    def reset_alert(self):
        self.alert_triggered = False
        self.is_hand_open_state = False
        self.hand_open_frames = 0
        self.hand_closed_frames = 0
        logger.info("Alerta resetado com sucesso!")

    def _init_tasks_api(self):
        try:
            from mediapipe.tasks.python import vision
            import os

            self.vision = vision

            model_path = "pose_landmarker_full.task" if os.path.exists("pose_landmarker_full.task") else "pose_landmarker_lite.task"
            logger.info(f"Usando modelo: {model_path}")

            base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
            self.options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )

            self.detector = vision.PoseLandmarker.create_from_options(self.options)
            self.ready = True
            self.last_timestamp_ms = 0
            logger.info("MediaPipe PoseLandmarker inicializado com sucesso!")

        except Exception as e:
            logger.error(f"Erro ao inicializar MediaPipe Tasks API: {e}")
            import traceback
            traceback.print_exc()
            self.ready = False

    def get_bounding_box(self, landmarks, image_shape):
        if not landmarks:
            return None

        h, w, _ = image_shape
        x_coords = [lm.x * w for lm in landmarks]
        y_coords = [lm.y * h for lm in landmarks]

        if not x_coords or not y_coords:
            return None

        x_min = int(max(0, min(x_coords) - 20))
        y_min = int(max(0, min(y_coords) - 20))
        x_max = int(min(w, max(x_coords) + 20))
        y_max = int(min(h, max(y_coords) + 20))

        return [x_min, y_min, x_max, y_max]

    def check_arm_raised(self, landmarks):
        if not landmarks or len(landmarks) < 17:
            return False

        try:
            nose       = landmarks[0]
            r_shoulder = landmarks[12]
            r_elbow    = landmarks[14]
            r_wrist    = landmarks[16]
            l_shoulder = landmarks[11]
            l_elbow    = landmarks[13]
            l_wrist    = landmarks[15]

            logger.debug(
                f"[POSE] R_Wrist={r_wrist.y:.2f} R_Elbow={r_elbow.y:.2f} R_Shoulder={r_shoulder.y:.2f} | "
                f"L_Wrist={l_wrist.y:.2f} L_Elbow={l_elbow.y:.2f} L_Shoulder={l_shoulder.y:.2f} | "
                f"Nose={nose.y:.2f}"
            )

            def arm_is_raised(wrist, elbow, shoulder):
                return (
                    wrist.y < (shoulder.y - 0.10) and
                    elbow.y < (shoulder.y + 0.05) and
                    wrist.y < (nose.y + 0.10)
                )

            right_raised = arm_is_raised(r_wrist, r_elbow, r_shoulder)
            left_raised  = arm_is_raised(l_wrist, l_elbow, l_shoulder)

            if right_raised or left_raised:
                side = "direito" if right_raised else "esquerdo"
                logger.warning(f"Braço {side} levantado detectado!")
                return True

        except IndexError:
            pass

        return False

    def process(self, frame):
        result = {
            "people": [],
            "alert_triggered": self.alert_triggered,
            "hand_open": False
        }

        if not self.ready:
            return result

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            current_timestamp_ms = int(time.time() * 1000)
            if self.last_timestamp_ms >= current_timestamp_ms:
                current_timestamp_ms = self.last_timestamp_ms + 1
            self.last_timestamp_ms = current_timestamp_ms

            detection_result = self.detector.detect_for_video(mp_img, current_timestamp_ms)

            hand_open_current = False
            num_detected = len(detection_result.pose_landmarks) if detection_result.pose_landmarks else 0
            if num_detected == 0:
                logger.debug("Nenhum esqueleto detectado no frame atual.")
            else:
                logger.debug(f"{num_detected} esqueleto(s) detectado(s).")

            if detection_result.pose_landmarks:
                for idx, landmarks in enumerate(detection_result.pose_landmarks):
                    box = self.get_bounding_box(landmarks, frame.shape)
                    if box:
                        result["people"].append({
                            "track_id": idx + 1,
                            "box": box,
                            "landmarks": landmarks
                        })
                        if self.check_arm_raised(landmarks):
                            hand_open_current = True

            if not self.alert_triggered:
                if hand_open_current:
                    self.hand_open_frames += 1
                    self.hand_closed_frames = 0
                    logger.info(f"Frames válidos: {self.hand_open_frames}/{self.frames_required_open}")
                    if self.hand_open_frames >= self.frames_required_open:
                        if not self.is_hand_open_state:
                            logger.info("ALERTA PERMANENTE ATIVADO!")
                            self.is_hand_open_state = True
                            self.alert_triggered = True
                else:
                    self.hand_closed_frames += 1
                    self.hand_open_frames = 0

            result["hand_open"] = self.is_hand_open_state or self.alert_triggered
            result["alert_triggered"] = self.alert_triggered

        except Exception as e:
            logger.error(f"Erro ao processar frame: {e}")
            import traceback
            traceback.print_exc()

        return result

    def close(self):
        if self.ready and hasattr(self, 'detector'):
            self.detector.close()
            self.ready = False
