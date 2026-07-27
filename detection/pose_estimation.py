from utils.logger import sys_logger

class PoseEstimator:
    """
    Avalia e valida poses proibidas com buffer de confirmação por quadros consecutivos (frames).
    """
    def __init__(self, required_consecutive_frames: int = 5):
        self.required_consecutive_frames = required_consecutive_frames
        self.frame_counters = {}  # (track_id, pose_name) -> int
        self.forbidden_poses = {
            "ARM_RAISED": self._check_arm_raised,
            "HANDS_UP": self._check_hands_up,
            "FALLEN": self._check_fallen
        }

    def evaluate(self, landmarks, track_id: int = 1):
        """
        Avalia os landmarks contra todas as poses proibidas.
        Exige confirmação de N frames consecutivos para a mesma pessoa antes de validar o evento.
        """
        if not landmarks:
            return False, None

        for pose_name, rule_fn in self.forbidden_poses.items():
            try:
                key = (track_id, pose_name)
                is_detected = rule_fn(landmarks)

                if is_detected:
                    current_count = self.frame_counters.get(key, 0) + 1
                    self.frame_counters[key] = current_count
                    if current_count >= self.required_consecutive_frames:
                        return True, pose_name
                else:
                    # Reseta o contador para 0 se o quadro não mantiver a condição
                    self.frame_counters[key] = 0
            except Exception as e:
                sys_logger.error(f"Erro ao avaliar pose '{pose_name}': {e}")

        return False, None

    def _check_arm_raised(self, landmarks):
        if len(landmarks) < 17:
            return False
        r_shoulder, r_elbow, r_wrist = landmarks[12], landmarks[14], landmarks[16]
        l_shoulder, l_elbow, l_wrist = landmarks[11], landmarks[13], landmarks[15]

        # Em coordenadas normalizadas: Y=0 topo, Y=1 base.
        # Braço erguido exige pulso claramente acima do ombro (diferença >= 0.08) e acima do cotovelo
        right_up = (r_wrist.y < (r_shoulder.y - 0.08)) and (r_wrist.y < (r_elbow.y - 0.04))
        left_up = (l_wrist.y < (l_shoulder.y - 0.08)) and (l_wrist.y < (l_elbow.y - 0.04))

        return right_up or left_up

    def _check_hands_up(self, landmarks):
        if len(landmarks) < 17:
            return False
        r_shoulder, r_wrist = landmarks[12], landmarks[16]
        l_shoulder, l_wrist = landmarks[11], landmarks[15]

        return (r_wrist.y < (r_shoulder.y - 0.10)) and (l_wrist.y < (l_shoulder.y - 0.10))

    def _check_fallen(self, landmarks):
        if len(landmarks) < 25:
            return False
        nose = landmarks[0]
        l_hip, r_hip = landmarks[23], landmarks[24]
        avg_hip_y = (l_hip.y + r_hip.y) / 2.0

        # Para queda: o corpo deve estar horizontal E a pessoa deve estar na parte inferior do frame (chão)
        is_horizontal = abs(nose.y - avg_hip_y) < 0.10
        is_on_floor = avg_hip_y > 0.70

        return is_horizontal and is_on_floor
