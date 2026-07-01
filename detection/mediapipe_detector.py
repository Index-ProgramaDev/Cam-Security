
import cv2
import numpy as np
import mediapipe as mp
import time
from loguru import logger


class MediaPipeDetector:
    def __init__(self, model_path="holistic_landmarker.task"):
        self._init_tasks_api(model_path)

        self.next_track_id = 1
        self.current_track_id = None
        self.lost_frames = 0
        self.max_lost_frames = 30
        
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

    def _init_tasks_api(self, model_path):
        try:
            from mediapipe.tasks.python import vision

            self.vision = vision

            base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
            self.options = vision.HolisticLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                min_face_detection_confidence=0.6,
                min_face_suppression_threshold=0.6,
                min_face_landmarks_confidence=0.6,
                min_pose_detection_confidence=0.6,
                min_pose_suppression_threshold=0.6,
                min_pose_landmarks_confidence=0.6,
                min_hand_landmarks_confidence=0.8,
                output_face_blendshapes=False,
                output_segmentation_mask=False
            )

            self.detector = vision.HolisticLandmarker.create_from_options(self.options)
            self.ready = True
            self.last_timestamp_ms = 0
            logger.info("MediaPipe HolisticLandmarker inicializado com sucesso!")

        except Exception as e:
            logger.error(f"Erro ao inicializar MediaPipe Tasks API: {e}")
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

    def check_hand_open(self, hand_landmarks):
        if not hand_landmarks or len(hand_landmarks) <21:
            return False
        
        open_count = 0
        
        thumb_tip = hand_landmarks[4]
        thumb_ip = hand_landmarks[3]
        thumb_mcp = hand_landmarks[2]
        
        if self.is_thumb_extended_simple(thumb_tip, thumb_ip, thumb_mcp):
            open_count += 1
        
        finger_pairs = [
            (8,6,5),
            (12,10,9),
            (16,14,13),
            (20,18,17)
        ]
        
        for tip_idx, pip_idx, mcp_idx in finger_pairs:
            tip = hand_landmarks[tip_idx]
            pip = hand_landmarks[pip_idx]
            mcp = hand_landmarks[mcp_idx]
            
            if self.is_finger_extended(tip, pip, mcp):
                open_count +=1
        
        return open_count >=5

    def process(self, frame):
        result = {
            "track_id": None,
            "box": None,
            "pose_landmarks": None,
            "face_landmarks": None,
            "left_hand_landmarks": None,
            "right_hand_landmarks": None,
            "hand_open": False,
            "alert_triggered": self.alert_triggered,
            "detected": False
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

            if detection_result.pose_landmarks:
                result["detected"] = True
                result["pose_landmarks"] = detection_result.pose_landmarks
                result["box"] = self.get_bounding_box(detection_result.pose_landmarks, frame.shape)

                if self.current_track_id is None:
                    self.current_track_id = self.next_track_id
                    self.next_track_id += 1
                    logger.info(f"Novo tracking ID: {self.current_track_id}")
                self.lost_frames = 0
                result["track_id"] = self.current_track_id
            else:
                self.lost_frames += 1
                if self.lost_frames > self.max_lost_frames:
                    self.current_track_id = None
                    self.lost_frames = 0

            if detection_result.face_landmarks:
                result["face_landmarks"] = detection_result.face_landmarks

            hand_open_current = False
            if detection_result.left_hand_landmarks:
                result["left_hand_landmarks"] = detection_result.left_hand_landmarks
                if self.check_hand_open(detection_result.left_hand_landmarks):
                    hand_open_current = True
            
            if detection_result.right_hand_landmarks:
                result["right_hand_landmarks"] = detection_result.right_hand_landmarks
                if self.check_hand_open(detection_result.right_hand_landmarks):
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
        if self.ready and hasattr(self, 'detector'):
            self.detector.close()
            self.ready = False

