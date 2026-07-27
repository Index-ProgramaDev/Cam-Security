import cv2
import threading
import time
import os
import yaml
from utils.logger import sys_logger

class CameraCapture:
    def __init__(self, config_path="config/config_camera.yaml"):
        self.config_path = config_path
        self.width = 640
        self.height = 480
        self.fps = 30
        self.camera_id = 0
        self.load_config()

        self.cap = None
        self.running = False
        self.frame = None
        self.lock = threading.Lock()
        self.thread = None

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
            except Exception as e:
                sys_logger.error(f"Erro ao carregar configuração de câmera: {e}")

    def start(self):
        if self.running:
            return

        sys_logger.info(f"Inicializando câmera física (ID: {self.camera_id})...")
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            # Tentar id 0 a 3 caso o ID configurado falhe
            for alt_id in range(4):
                if alt_id == self.camera_id:
                    continue
                sys_logger.info(f"Tentando câmera ID {alt_id}...")
                self.cap = cv2.VideoCapture(alt_id)
                if self.cap.isOpened():
                    self.camera_id = alt_id
                    break

        if not self.cap.isOpened():
            sys_logger.error("Nenhuma câmera física detectada!")
            raise RuntimeError("Não foi possível conectar a uma câmera física.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.running = True

        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        sys_logger.info("Câmera inicializada com sucesso.")

    def _read_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        sys_logger.info("Câmera encerrada.")
