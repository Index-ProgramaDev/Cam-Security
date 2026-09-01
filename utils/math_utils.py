import math
import numpy as np

DIST_PERTO = 1.0
DIST_MEDIO = 3.0
PUNCH_VELOCITY_THRESHOLD = 500.0  # px/s
PUNCH_ANGLE_THRESHOLD = 150.0     # graus (cotovelo quase estendido)


def calculate_center(box):
    x1, y1, x2, y2 = box[:4]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def calculate_relative_distance(box1, box2):
    """Distância entre centros normalizada pela altura média. Retorna (dist, label)."""
    cx1, cy1 = calculate_center(box1)
    cx2, cy2 = calculate_center(box2)
    dist_px = math.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)
    avg_h = (abs(box1[3] - box1[1]) + abs(box2[3] - box2[1])) / 2.0
    if avg_h <= 0:
        return 999.0, "LONGE"
    rel = dist_px / avg_h
    label = "PERTO" if rel < DIST_PERTO else ("MEDIO" if rel <= DIST_MEDIO else "LONGE")
    return rel, label


def calculate_velocity(pos_atual, pos_anterior, delta_tempo):
    """Velocidade em pixels/segundo entre dois pontos (x, y)."""
    if delta_tempo <= 0 or pos_atual is None or pos_anterior is None:
        return 0.0
    dx = pos_atual[0] - pos_anterior[0]
    dy = pos_atual[1] - pos_anterior[1]
    return math.sqrt(dx * dx + dy * dy) / delta_tempo


def detect_collision(box1, box2, iou_threshold=0.05):
    """True se IoU entre as duas caixas exceder o threshold."""
    xA, yA = max(box1[0], box2[0]), max(box1[1], box2[1])
    xB, yB = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return False
    areaA = (box1[2] - box1[0]) * (box1[3] - box1[1])
    areaB = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return (inter / float(areaA + areaB - inter)) > iou_threshold


def calculate_arm_angle(shoulder, elbow, wrist):
    """Ângulo em graus na articulação do cotovelo."""
    if not (shoulder and elbow and wrist):
        return 0.0
    a = np.array([shoulder.x, shoulder.y])
    b = np.array([elbow.x, elbow.y])
    c = np.array([wrist.x, wrist.y])
    ba, bc = a - b, c - b
    n_ba, n_bc = np.linalg.norm(ba), np.linalg.norm(bc)
    if n_ba == 0 or n_bc == 0:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / (n_ba * n_bc), -1.0, 1.0))))


def detect_punch(wrist_velocity, arm_angle, has_multiple_people, is_near_or_collision,
                 vel_threshold=PUNCH_VELOCITY_THRESHOLD, angle_threshold=PUNCH_ANGLE_THRESHOLD):
    """
    Detecta soco. Requer obrigatoriamente 2+ pessoas na cena e
    proximidade/colisão — nunca dispara para pessoa sozinha.
    """
    if not has_multiple_people:
        return False
    return (wrist_velocity >= vel_threshold
            and arm_angle >= angle_threshold
            and is_near_or_collision)


def calculate_risk_score(event_type: str, dist_label: str = "LONGE",
                         velocity: float = 0.0, has_collision: bool = False) -> int:
    """Risk score 0-100 baseado em tipo de evento, distância, velocidade e colisão."""
    base = {"SOCO": 90, "FALLEN": 80, "HANDS_UP": 50, "ARM_RAISED": 30, "COLISAO": 40}
    score = base.get(event_type, 20)
    if has_collision:
        score += 20
    if dist_label == "PERTO":
        score += 15
    elif dist_label == "MEDIO":
        score += 5
    if velocity > 300.0:
        score += 15
    elif velocity > 150.0:
        score += 5
    return int(min(100, max(0, score)))
