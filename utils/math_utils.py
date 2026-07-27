import math
import numpy as np

# Constantes configuráveis
DIST_PERTO = 1.0
DIST_MEDIO = 3.0
PUNCH_VELOCITY_THRESHOLD = 500.0  # px/s (movimento brusco)
PUNCH_ANGLE_THRESHOLD = 150.0     # graus (braço quase 180° estendido)

def calculate_center(box):
    """Retorna o centro (cx, cy) de uma bbox [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = box[:4]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def calculate_relative_distance(box1, box2):
    """
    Calcula a distância relativa entre duas caixas distintas normalizada pela altura média.
    Retorna (rel_dist, 'PERTO' | 'MEDIO' | 'LONGE').
    """
    cx1, cy1 = calculate_center(box1)
    cx2, cy2 = calculate_center(box2)
    dist_px = math.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)

    h1 = abs(box1[3] - box1[1])
    h2 = abs(box2[3] - box2[1])
    avg_h = (h1 + h2) / 2.0

    if avg_h <= 0:
        return 999.0, "LONGE"

    rel_dist = dist_px / avg_h

    if rel_dist < DIST_PERTO:
        label = "PERTO"
    elif rel_dist <= DIST_MEDIO:
        label = "MEDIO"
    else:
        label = "LONGE"

    return rel_dist, label

def calculate_velocity(pos_atual, pos_anterior, delta_tempo):
    """Calcula velocidade em pixels/segundo entre dois pontos (x, y)."""
    if delta_tempo <= 0 or pos_atual is None or pos_anterior is None:
        return 0.0
    dx = pos_atual[0] - pos_anterior[0]
    dy = pos_atual[1] - pos_anterior[1]
    dist = math.sqrt(dx*dx + dy*dy)
    return dist / delta_tempo

def detect_collision(box1, box2, iou_threshold=0.05):
    """Retorna True se houver colisão/sobreposição de IoU > threshold entre duas caixas distintas."""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return False

    boxAArea = (box1[2] - box1[0]) * (box1[3] - box1[1])
    boxBArea = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou > iou_threshold

def calculate_arm_angle(shoulder, elbow, wrist):
    """Calcula o ângulo em graus na articulação do cotovelo."""
    if not (shoulder and elbow and wrist):
        return 0.0
    a = np.array([shoulder.x, shoulder.y])
    b = np.array([elbow.x, elbow.y])
    c = np.array([wrist.x, wrist.y])

    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return float(np.degrees(angle))

def detect_punch(wrist_velocity, arm_angle, has_multiple_people, is_near_or_collision, vel_threshold=PUNCH_VELOCITY_THRESHOLD, angle_threshold=PUNCH_ANGLE_THRESHOLD):
    """
    EXIGE EXPLICITAMENTE pelo menos 2 pessoas rastreadas na cena.
    Se só houver 1 pessoa, SOCO NUNCA dispara (retorna False).
    """
    if not has_multiple_people:
        return False

    fast_wrist = wrist_velocity >= vel_threshold
    arm_extended = arm_angle >= angle_threshold
    return fast_wrist and arm_extended and is_near_or_collision

def calculate_risk_score(event_type: str, dist_label: str = "LONGE", velocity: float = 0.0, has_collision: bool = False) -> int:
    """Calcula o Risk Score dinâmico baseado em evento, distância, velocidade e colisão."""
    base_scores = {
        "SOCO": 90,
        "FALLEN": 80,
        "HANDS_UP": 50,
        "ARM_RAISED": 30,
        "COLISAO": 40
    }
    score = base_scores.get(event_type, 20)

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