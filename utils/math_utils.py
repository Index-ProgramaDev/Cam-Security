import math

# Calculos especiais

# Distância euclidiana

def Distancia(x1, x2, y1, y2):
    return math.sqrt((x2 - x1)**2 + (y1 - y2)**2)

# Centro bounding da boxe

def calcular_centro(box):
    x, y, w, h = box
    cx = (x+ w) / 2
    cy = (y + h) / 2
    return (cx. cy)

# Velocidade

def calcular_velocidade(p_anterior, p_atual, tempo):
    espaco = calcular_distancia(p_anterior, p_atual)
    return espaco / tempo if tempo > 0 else 0

# Direção

def calcular_direcao(p_anterior, p_atual):
    dx = p_atual[0] - p_anterior[0]
    dy = p_atual[1] - p_anterior[1]

    magnitude = math.sqrt(dx**2 + dy**2)

    if magnitude == 0:
        return (0, 0)

    return (dx / magnitude, dy / magnitude)

# Colisão (Não testado)

def detectar_colisao(p1, p2, limite):
    distancia = calcular_distancia(p1, p2)
    return distancia < limite

# Verificar inatividade 

def esta_parado(velocidade, limite=1.0):
    return velocidade < limite