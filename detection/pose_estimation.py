from utils.logger import sys_logger

class PoseEstimator:
    def __init__(self):
        self.forbidden_poses = {
            "ARM_RAISED": self._check_arm_raised,
            "HANDS_UP": self._check_hands_up,
            "FALLEN": self._check_fallen
        }

    def add_forbidden_pose(self, name: str, rule_func):
        self.forbidden_poses[name] = rule_func
        sys_logger.info(f"Nova pose proibida cadastrada: {name}")

    def evaluate(self, landmarks):
        if not landmarks:
            return False, None

        for pose_name, rule_fn in self.forbidden_poses.items():
            try:
                if rule_fn(landmarks):
                    return True, pose_name
            except Exception as e:
                sys_logger.error(f"Erro ao avaliar pose '{pose_name}': {e}")
        return False, None

    def _check_arm_raised(self, landmarks):
        if len(landmarks) < 17:
            return False
        r_shoulder, r_elbow, r_wrist = landmarks[12], landmarks[14], landmarks[16]
        l_shoulder, l_elbow, l_wrist = landmarks[11], landmarks[13], landmarks[15]

        # Em coordenadas normalizadas, Y=0 topo, Y=1 base.
        # Braço erguido se pulso ou cotovelo estiver acima (menor Y) que o ombro.
        right_up = (r_wrist.y < r_shoulder.y) or (r_elbow.y < r_shoulder.y)
        left_up = (l_wrist.y < l_shoulder.y) or (l_elbow.y < l_shoulder.y)

        return right_up or left_up

    def _check_hands_up(self, landmarks):
        if len(landmarks) < 17:
            return False
        r_shoulder, r_wrist = landmarks[12], landmarks[16]
        l_shoulder, l_wrist = landmarks[11], landmarks[15]

        return (r_wrist.y < r_shoulder.y) and (l_wrist.y < l_shoulder.y)

    def _check_fallen(self, landmarks):
        if len(landmarks) < 25:
            return False
        nose = landmarks[0]
        l_hip, r_hip = landmarks[23], landmarks[24]
        avg_hip_y = (l_hip.y + r_hip.y) / 2.0
        return abs(nose.y - avg_hip_y) < 0.15
