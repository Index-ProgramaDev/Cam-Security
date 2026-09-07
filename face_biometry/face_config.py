"""
face_config.py — fonte única de configuração da camada facial.

Todos os módulos do pipeline facial devem importar daqui.
Não espalhe valores mágicos como 128 pelo projeto.
"""

# Dimensão do vetor de embedding gerado pelo FaceDetector (HOG 4×4×8).
# Contrato rígido: qualquer embedding com dimensão diferente deve ser REJEITADO.
EMBEDDING_DIM: int = 128

# Intervalo mínimo entre capturas de embedding válidas para o mesmo track (segundos).
EMBEDDING_CAPTURE_INTERVAL: float = 3.0

# Número máximo de embeddings acumulados por track durante uma sessão/evento.
# Após atingir o limite, nenhum novo embedding é coletado para aquele track.
MAX_EMBEDDINGS_PER_TRACK: int = 8
