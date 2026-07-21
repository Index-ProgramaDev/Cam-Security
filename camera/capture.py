import cv2
import threading
import time
import os
import yaml
import numpy as np
from loguru import logger
        
class CameraCapture:

    def __init__(self, config_path="config/config_camera.yaml"):

        # Configuração de Câmera

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
