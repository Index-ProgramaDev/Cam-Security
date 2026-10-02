from utils.logger import sys_logger


class PoseEstimator:
    """
    Avalia poses proibidas com confirmação por N frames consecutivos por track.
    Evita falsos positivos causados por detecções momentâneas.
    """

    def __init__(self, required_consecutive_frames: int = 5):
        self.required_consecutive_frames = required_consecutive_frames
        self.frame_counters = {}  # (track_id, pose_name) -> int
        self.forbidden_poses = {
            "HANDS_UP":   self._check_hands_up,
        }

    def evaluate(self, landmarks, track_id: int = 1):
        """
        Retorna (True, pose_name) quando uma pose proibida for confirmada por
        required_consecutive_frames quadros. Caso contrário, retorna (False, None).
        """
        if not landmarks:
            return False, None

        for pose_name, rule_fn in self.forbidden_poses.items():
            try:
                key = (track_id, pose_name)
                if rule_fn(landmarks):
                    count = self.frame_counters.get(key, 0) + 1
                    self.frame_counters[key] = count
                    if count >= self.required_consecutive_frames:
                        return True, pose_name
                else:
                    self.frame_counters[key] = 0
            except Exception as e:
                sys_logger.error(f"Erro ao avaliar pose '{pose_name}': {e}")

        return False, None

    def _check_arm_raised(self, landmarks):
        if len(landmarks) < 17:
            return False
        r_shoulder, r_elbow, r_wrist = landmarks[12], landmarks[14], landmarks[16]
        l_shoulder, l_elbow, l_wrist = landmarks[11], landmarks[13], landmarks[15]
        right_up = (r_wrist.y < r_shoulder.y - 0.08) and (r_wrist.y < r_elbow.y - 0.04)
        left_up  = (l_wrist.y < l_shoulder.y - 0.08) and (l_wrist.y < l_elbow.y - 0.04)
        return right_up or left_up

    def _check_hands_up(self, landmarks):
        if len(landmarks) < 17:
            return False
        r_shoulder, r_wrist = landmarks[12], landmarks[16]
        l_shoulder, l_wrist = landmarks[11], landmarks[15]
        return (r_wrist.y < r_shoulder.y - 0.10) and (l_wrist.y < l_shoulder.y - 0.10)


