
import cv2
import numpy as np
import mediapipe as mp
import time
from loguru import logger


class MediaPipeDetector:
    def __init__(self):
        self._init_old_mediapipe()
        
        self.hand_open_frames = 0
        self.hand_closed_frames = 0
        self.is_hand_open_state = False
        self.frames_required_open = 8
        
        self.alert_triggered = False

    def reset_alert(self):
        self.alert_triggered = False
        self.is_hand_open_state = False
        self.hand_open_frames = 0
        self.hand_closed_frames = 0
        logger.info("Alerta resetado com sucesso!")

    def _init_old_mediapipe(self):
        try:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
            self.ready = True
            logger.info("MediaPipe Pose (API antiga) inicializado com sucesso!")
        except Exception as e:
            logger.error(f"Erro ao inicializar MediaPipe Pose: {e}")
            import traceback
            traceback.print_exc()
            self.ready = False

    def get_bounding_box(self, landmarks, image_shape):
        if not landmarks:
            return None

        h, w, _ = image_shape
        x_coords = []
        y_coords = []

        for lm in landmarks:
            x_coords.append(lm.x * w)
            y_coords.append(lm.y * h)

        if not x_coords or not y_coords:
            return None

        x_min = int(max(0, min(x_coords) - 20))
        y_min = int(max(0, min(y_coords) - 20))
        x_max = int(min(w, max(x_coords) + 20))
        y_max = int(min(h, max(y_coords) + 20))

        return [x_min, y_min, x_max, y_max]

    def is_finger_extended(self, finger_tip, finger_pip, finger_mcp):
        return finger_tip.y < finger_pip.y and finger_tip.y < finger_mcp.y

    def is_thumb_extended_simple(self, thumb_tip, thumb_ip, thumb_mcp):
        wrist = thumb_mcp
        distance_tip = ((thumb_tip.x - wrist.x)**2 + (thumb_tip.y - wrist.y)**2)**0.5
        distance_ip = ((thumb_ip.x - wrist.x)**2 + (thumb_ip.y - wrist.y)**2)**0.5
        
        return distance_tip > distance_ip * 1.3

    def check_hand_open(self, landmarks):
        if not landmarks or len(landmarks) < 21:
            return False
        
        open_count = 0
        
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        
        if self.is_thumb_extended_simple(thumb_tip, thumb_ip, thumb_mcp):
            open_count += 1
        
        finger_pairs = [
            (8,6,5),
            (12,10,9),
            (16,14,13),
            (20,18,17)
        ]
        
        for tip_idx, pip_idx, mcp_idx in finger_pairs:
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]
            mcp = landmarks[mcp_idx]
            
            if self.is_finger_extended(tip, pip, mcp):
                open_count +=1
        
        return open_count >=5

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
            results = self.pose.process(rgb_frame)

            hand_open_current = False
            if results.pose_landmarks:
                box = self.get_bounding_box(results.pose_landmarks.landmark, frame.shape)
                if box:
                    person_data = {
                        "track_id": 1,
                        "box": box,
                        "landmarks": results.pose_landmarks
                    }
                    result["people"].append(person_data)

                    if self.check_hand_open(results.pose_landmarks.landmark):
                        hand_open_current = True

            if not self.alert_triggered:
                if hand_open_current:
                    self.hand_open_frames += 1
                    self.hand_closed_frames = 0
                    if self.hand_open_frames >= self.frames_required_open:
                        if not self.is_hand_open_state:
                            logger.info("MÃO ABERTA DETECTADA! ALERTA PERMANENTE ATIVADO!")
                            self.is_hand_open_state = True
                            self.alert_triggered = True
                else:
                    self.hand_closed_frames +=1
                    self.hand_open_frames = 0
            
            result["hand_open"] = self.is_hand_open_state or self.alert_triggered
            result["alert_triggered"] = self.alert_triggered

        except Exception as e:
            logger.error(f"Erro ao processar frame: {e}")
            import traceback
            traceback.print_exc()

        return result

    def close(self):
        pass

