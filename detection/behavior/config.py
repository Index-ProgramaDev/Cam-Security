# ==========================================
# CONFIGURAÇÕES DA CAMADA COMPORTAMENTAL
# ==========================================
# Todos os valores de velocidade/distância são NORMALIZADOS
# (divididos pela altura média das bounding boxes, em unidades/segundo).

# Histórico Temporal
HISTORY_WINDOW_SECONDS   = 2.0   # Janela máxima de histórico retida por track
HISTORY_CLEANUP_INTERVAL = 5.0   # Frequência de limpeza de tracks inativos (s)

# ------------------------------------------
# Sinais Base (nunca viram AlertManager)
# ------------------------------------------
HIGH_SPEED_THRESHOLD       = 2.5  # vel. normalizada mínima → sinal high_speed
SUDDEN_MOVEMENT_THRESHOLD  = 3.5  # aceleração normalizada mínima → sinal sudden_movement
MIN_STABLE_HISTORY         = 5    # snapshots mínimos para aceitar qualquer sinal

# ------------------------------------------
# SOCO (PUNCH)
# ------------------------------------------
PUNCH_MIN_CONFIDENCE    = 0.72
PUNCH_MIN_WRIST_SPEED   = 2.2    # Velocidade mínima do punho (norm./s)
PUNCH_MIN_ACCELERATION  = 5.0    # Aceleração mínima do punho  (norm./s²)
PUNCH_MIN_EXTENSION     = 0.40   # Extensão mínima ombro→punho (norm. pela largura bbox)
PUNCH_COOLDOWN          = 3.5    # Cooldown entre socos do mesmo track (s)
PUNCH_WINDOW            = 0.6    # Janela temporal da sequência do soco (s)

# ------------------------------------------
# LUTA (FIGHT)
# ------------------------------------------
FIGHT_MIN_CONFIDENCE = 0.75
FIGHT_MAX_DISTANCE   = 2.0   # Distância máxima normalizada para interação
FIGHT_WINDOW         = 2.0   # Janela de persistência (s)
FIGHT_MIN_FRAMES     = 4     # Frames mínimos de interação contínua
FIGHT_COOLDOWN       = 6.0

# ------------------------------------------
# PERSEGUIÇÃO (PURSUIT)
# ------------------------------------------
PURSUIT_MIN_CONFIDENCE  = 0.78
PURSUIT_MIN_SPEED       = 1.2    # Velocidade mínima de AMBAS as pessoas
PURSUIT_MIN_DISTANCE    = 0.8    # Dist. norm. mínima (abaixo = andando juntos)
PURSUIT_MAX_DISTANCE    = 6.0    # Dist. norm. máxima (acima = sem relação)
PURSUIT_SIMILAR_SPEED   = 0.25   # Diferença máxima de vel. para considerar "amigos"
PURSUIT_WINDOW          = 2.0    # Janela temporal para calcular tendência (s)
PURSUIT_MAX_TRACKS      = 4      # Máximo de tracks na cena para ativar
PURSUIT_COOLDOWN        = 6.0

# ------------------------------------------
# COLISÃO (COLLISION)
# ------------------------------------------
COLLISION_MIN_CONFIDENCE  = 0.78
COLLISION_RELATIVE_SPEED  = 2.5   # Vel. relativa mínima (norm.)
COLLISION_CLOSE_THRESHOLD = 0.8   # Dist. norm. para considerar "muito perto"
COLLISION_COOLDOWN        = 4.0
