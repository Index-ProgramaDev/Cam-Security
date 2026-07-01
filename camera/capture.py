import cv2
import threading
import time
import os
import yaml
import numpy as np
from loguru import logger

class SyntheticCamera:
    """
    Fallback camera simulation that draws moving agents on a canvas.
    Ensures the application is runnable and testable without a physical camera.
    """
    def __init__(self, width=640, height=480, fps=20):
        self.width = width
        self.height = height
        self.fps = fps
        self.start_time = time.time()
        # Cada agente tem: id, cor, x, y, vx, vy, rx (width), ry (height), state
        # Aumentamos um pouco o tamanho das elipses para simular melhor o corpo humano
        self.agents = [
            {"id": 1, "color": (70, 70, 220), "x": 100.0, "y": 300.0, "vx": 1.2, "vy": 0.0, "rx": 22, "ry": 60, "state": "walking"},
            {"id": 2, "color": (70, 220, 70), "x": 50.0, "y": 200.0, "vx": 4.8, "vy": 0.2, "rx": 20, "ry": 55, "state": "running"},
            {"id": 3, "color": (220, 70, 70), "x": 380.0, "y": 240.0, "vx": 0.8, "vy": 0.0, "rx": 22, "ry": 60, "state": "walking"},
            {"id": 4, "color": (70, 220, 220), "x": 180.0, "y": 340.0, "vx": 1.6, "vy": -0.1, "rx": 22, "ry": 60, "state": "walking"},
            {"id": 5, "color": (220, 70, 220), "x": 480.0, "y": 350.0, "vx": -1.8, "vy": 0.1, "rx": 22, "ry": 60, "state": "walking"}
        ]
        
    def read(self):
        # Delay de FPS
        time.sleep(1.0 / self.fps)
        
        # Criar canvas de fundo
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Fundo de escritório cinza/azul escuro
        frame[:] = (35, 30, 30) # OpenCV usa BGR
        
        # Desenhar cenário (grid/linhas de perspectiva)
        cv2.line(frame, (0, 150), (self.width, 150), (70, 70, 70), 2)
        cv2.line(frame, (0, 420), (self.width, 420), (70, 70, 70), 2)
        
        # Desenhar portas/colunas
        cv2.rectangle(frame, (100, 60), (160, 150), (45, 45, 50), -1)
        cv2.rectangle(frame, (100, 60), (160, 150), (90, 90, 95), 2)
        cv2.rectangle(frame, (480, 60), (540, 150), (45, 45, 50), -1)
        cv2.rectangle(frame, (480, 60), (540, 150), (90, 90, 95), 2)
        
        # Desenhar uma placa "ÁREA MONITORADA"
        cv2.rectangle(frame, (250, 15), (390, 45), (0, 0, 120), -1)
        cv2.rectangle(frame, (250, 15), (390, 45), (0, 0, 200), 2)
        cv2.putText(frame, "CAM-SECURITY", (262, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Atualizar agentes
        t = time.time() - self.start_time
        
        for agent in self.agents:
            # Lógica para o Agente 3 cair (começa a cair aos 7 segundos)
            if agent["id"] == 3:
                if t > 7.0 and agent["state"] == "walking":
                    agent["state"] = "falling"
                    agent["vx"] = 0.3
                    agent["vy"] = 0.5
                if agent["state"] == "falling":
                    # Colapsando dimensões da elipse (ry diminui, rx aumenta)
                    agent["ry"] -= 4
                    agent["rx"] += 3
                    if agent["ry"] <= 18:
                        agent["ry"] = 18
                        agent["state"] = "fell"
                        agent["vx"] = 0.0
                        agent["vy"] = 0.0
                        
            # Lógica de colisão (Agentes 4 e 5 se aproximam e cruzam aos 12 segundos)
            if agent["id"] == 4 and t > 12.0 and agent["state"] == "walking":
                # Aumenta velocidade e muda direção em direção ao outro
                agent["vx"] = 3.0
                agent["vy"] = 0.2
            if agent["id"] == 5 and t > 12.0 and agent["state"] == "walking":
                agent["vx"] = -3.0
                agent["vy"] = -0.2

            # Mover agentes
            agent["x"] += agent["vx"]
            agent["y"] += agent["vy"]
            
            # Limites horizontais (wrap-around)
            if agent["x"] < -40:
                agent["x"] = self.width + 40
                if agent["state"] == "fell":
                    # Ressuscitar se der a volta
                    agent["state"] = "walking"
                    agent["rx"], agent["ry"] = 22, 60
                    agent["vx"], agent["vy"] = 0.8, 0.0
            elif agent["x"] > self.width + 40:
                agent["x"] = -40
                if agent["state"] == "fell":
                    agent["state"] = "walking"
                    agent["rx"], agent["ry"] = 22, 60
                    agent["vx"], agent["vy"] = 0.8, 0.0
            
            # Limites verticais
            if agent["y"] < 160:
                agent["y"] = 160
                agent["vy"] = abs(agent["vy"])
            elif agent["y"] > 410:
                agent["y"] = 410
                agent["vy"] = -abs(agent["vy"])
                
            # Desenhar corpo elíptico
            cx, cy = int(agent["x"]), int(agent["y"])
            rx, ry = int(agent["rx"]), int(agent["ry"])
            cv2.ellipse(frame, (cx, cy), (rx, ry), 0, 0, 360, agent["color"], -1)
            cv2.ellipse(frame, (cx, cy), (rx, ry), 0, 0, 360, (230, 230, 230), 1)
            
            # Desenhar cabeça
            if agent["state"] == "fell":
                hx, hy = cx + rx - 5, cy + 2
            else:
                hx, hy = cx, cy - ry - 12
            cv2.circle(frame, (int(hx), int(hy)), 11, (240, 200, 180), -1)
            cv2.circle(frame, (int(hx), int(hy)), 11, (230, 230, 230), 1)
            
        return True, frame
        
    def release(self):
        pass

class CameraCapture:
    """
    Thread-safe continuous camera capture that handles OpenCV initialization
    and provides a synthetic camera fallback.
    """
    def __init__(self, config_path="config/config_camera.yaml"):
        self.config_path = config_path
        self.width = 640
        self.height = 480
        self.fps = 30
        self.camera_id = 0
        self.use_synthetic = False
        
        self.load_config()
        
        self.cap = None
        self.running = False
        self.frame = None
        self.lock = threading.Lock()
        self.thread = None
        self.is_synthetic_active = False

    def load_config(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                    if cfg:
                        self.camera_id = cfg.get("camera_id", 0)
                        self.width = cfg.get("width", 640)
                        self.height = cfg.get("height", 480)
                        self.fps = cfg.get("fps", 30)
                        self.use_synthetic = cfg.get("use_synthetic", False)
            except Exception as e:
                logger.error(f"Erro ao ler configuração da câmera: {e}")

    def start(self):
        if self.running:
            return
            
        self.running = True
        
        # Tenta inicializar câmera física se não estiver forçado para sintética
        if not self.use_synthetic:
            logger.info("Procurando câmeras físicas disponíveis (IDs 0 a 3)...")
            opened = False
            
            # Tentar o ID configurado primeiro, e depois de 0 a 3
            ids_to_try = [self.camera_id]
            for i in range(4):
                if i not in ids_to_try:
                    ids_to_try.append(i)
                    
            for src in ids_to_try:
                src_val = src
                if isinstance(src_val, str) and src_val.isdigit():
                    src_val = int(src_val)
                    
                logger.info(f"Tentando inicializar captura física (ID: {src_val})...")
                # Abre a câmera sem backend específico primeiro
                self.cap = cv2.VideoCapture(src_val)
                if self.cap.isOpened():
                    self.camera_id = src_val
                    # Espera um pouco para a câmera inicializar antes de definir resolução
                    import time
                    time.sleep(0.5)
                    # Tenta definir resolução, mas se falhar, usa a resolução padrão
                    if not self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width):
                        logger.warning("Não foi possível definir a resolução desejada para a largura.")
                    if not self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height):
                        logger.warning("Não foi possível definir a resolução desejada para a altura.")
                    # Confirma a resolução final
                    final_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    final_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    logger.info(f"Câmera física ID {src_val} inicializada com sucesso (resolução: {final_width}x{final_height}).")
                    self.is_synthetic_active = False
                    opened = True
                    break
                else:
                    self.cap.release()
                    
            if not opened:
                logger.error("ERRO CRÍTICO: Nenhuma câmera física pôde ser aberta.")
                logger.error("Certifique-se de que sua webcam está conectada e não está sendo usada por outro app.")
                import sys
                self.running = False
                sys.exit(1)
        else:
            logger.info("Modo câmera sintética forçado nas configurações.")
            self.cap = SyntheticCamera(self.width, self.height, self.fps)
            self.is_synthetic_active = True
            
        # Iniciar thread de leitura
        self.thread = threading.Thread(target=self._read_loop, name="CameraCaptureThread", daemon=True)
        self.thread.start()

    def _read_loop(self):
        import traceback
        while self.running:
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    if not self.is_synthetic_active:
                        # Verifica se precisa redimensionar
                        h, w = frame.shape[:2]
                        if w != self.width or h != self.height:
                            frame = cv2.resize(frame, (self.width, self.height))
                    with self.lock:
                        self.frame = frame.copy()
                else:
                    if not self.is_synthetic_active:
                        # Se falhar, espera menos tempo e tenta de novo
                        time.sleep(0.033)  # ~30fps
                    else:
                        time.sleep(0.01)
            except Exception as e:
                logger.error(f"Erro no loop de leitura: {e}")
                traceback.print_exc()
                time.sleep(0.1)

    def get_frame(self):
        """Retorna o frame atual de forma thread-safe."""
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        logger.info("Captura de câmera parada.")