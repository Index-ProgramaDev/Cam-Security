import time

# Agora

def tempo_atual():
    return time.time()

# Delta do tempo pra calculo de velocidade

def tempo_delta(t1, t2):
    return t2 - t1


# Timer
class Timer:
    def __init__(self):
        self.start = tempo_atual()

    def reset(self):
        self.start = tempo_atual()

    def tempo_passado(self):
        return tempo_atual() - self.start
    
    # Expiração
    
def expirou(inicio, limite_segundos):
    return (tempo_atual() - inicio) > limite_segundos

# AntiSpam

def expirou(inicio, limite_segundos):
    return (tempo_atual() - inicio) > limite_segundos