
import cv2
import numpy as np
import mediapipe as mp
import time
from loguru import logger
from utils.math_utils import calcular_distancia, calcular_centro


class PersonData:
    def __init__(self, track_id, landmarks, box):
        self.track_id = track_id
        self.landmarks = landmarks
        self.box = box
        self.center = calcular_centro(box)
        self.lost_frames = 0


class MediaPipeDetector:
    def __init__(self, model_path="pose_landmarker_lite.task"):
        self._init_tasks_api(model_path)

        self.next_track_id = 1
        self.tracked_people = {}
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
            self.options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_poses=5,
                min_pose_detection_confidence=0.6,
                min_pose_presence_confidence=0.6,
                min_tracking_confidence=0.6
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

    def update_tracks(self, new_landmarks_list, image_shape):
        new_tracked = {}
        for landmarks in new_landmarks_list:
            box = self.get_bounding_box(landmarks, image_shape)
            if not box:
                continue
            center = calcular_centro(box)
            matched_id = None

            min_dist = float('inf')
            for track_id, person in self.tracked_people.items():
                dist = calcular_distancia(center, person.center)
                if dist < 100 and dist < min_dist:
                    min_dist = dist
                    matched_id = track_id

            if matched_id is not None:
                new_tracked[matched_id] = PersonData(matched_id, landmarks, box)
                self.tracked_people.pop(matched_id)
            else:
                new_track_id = self.next_track_id
                self.next_track_id += 1
                new_tracked[new_track_id] = PersonData(new_track_id, landmarks, box)
                logger.info(f"Novo tracking ID: {new_track_id}")

        for track_id, person in self.tracked_people.items():
            person.lost_frames += 1
            if person.lost_frames < self.max_lost_frames:
                new_tracked[track_id] = person

        self.tracked_people = new_tracked
        return list(self.tracked_people.values())

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
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            current_timestamp_ms = int(time.time() * 1000)
            if self.last_timestamp_ms >= current_timestamp_ms:
                current_timestamp_ms = self.last_timestamp_ms + 1
            self.last_timestamp_ms = current_timestamp_ms

            detection_result = self.detector.detect_for_video(mp_img, current_timestamp_ms)
            people = self.update_tracks(detection_result.pose_landmarks, frame.shape)

            hand_open_current = False
            for person in people:
                person_data = {
                    "track_id": person.track_id,
                    "box": person.box,
                    "center": person.center,
                    "landmarks": person.landmarks
                }
                result["people"].append(person_data)

                # Check hand open for any tracked person
                if self.check_hand_open(person.landmarks):
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

