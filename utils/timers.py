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
        self.start = agora()

    def reset(self):
        self.start = agora()

    def tempo_passado(self):
        return agora() - self.start
    
    # Expiração
    
def expirou(inicio, limite_segundos):
    return (agora() - inicio) > limite_segundos

# AntiSpam

def expirou(inicio, limite_segundos):
    return (agora() - inicio) > limite_segundos