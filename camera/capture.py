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
        self.video_path = ""
        self.is_video_source = False
        self.frame_interval = 0.0
        self.load_config()

        self.cap = None
        self.source = None
        self.running = False
        self.frame = None
        self.frame_seq = 0
        self.lock = threading.Lock()
        self.thread = None

    def load_config(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                    if cfg:
                        self.camera_id = cfg.get("camera_id", 0)
                        self.video_path = cfg.get("video_path", "")
                        self.width = cfg.get("width", 640)
                        self.height = cfg.get("height", 480)
                        self.fps = cfg.get("fps", 30)
                        self.frame_interval = 1.0 / max(self.fps, 1)
            except Exception as e:
                sys_logger.error(f"Erro ao carregar configuração de câmera: {e}")

    def start(self):
        if self.running:
            return

        source_to_open = None
        source_label = ""

        if self.video_path and os.path.exists(self.video_path):
            source_to_open = self.video_path
            source_label = f"vídeo ({self.video_path})"
            self.is_video_source = True
        elif isinstance(self.camera_id, str) and os.path.exists(self.camera_id):
            source_to_open = self.camera_id
            source_label = f"vídeo ({self.camera_id})"
            self.is_video_source = True
        else:
            source_label = f"câmera física (ID: {self.camera_id})"
            source_to_open = self.camera_id
            self.is_video_source = False

        sys_logger.info(f"Inicializando {source_label}...")
        self.source = source_to_open
        self.cap = cv2.VideoCapture(source_to_open)
        if self.is_video_source and self.cap.isOpened():
            real_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
            # Alguns containers reportam 0, 1000+ ou valores instáveis.
            if 1.0 <= real_fps <= 120.0:
                self.fps = real_fps
            self.frame_interval = 1.0 / max(self.fps, 1)
            sys_logger.info(f"Playback do vídeo em {self.fps:.2f} FPS.")
        else:
            self.frame_interval = 0.0

        if not self.cap.isOpened() and not isinstance(source_to_open, str):
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
            if self.video_path:
                sys_logger.error(f"Não foi possível abrir o vídeo: {self.video_path}")
                raise RuntimeError(f"Não foi possível abrir o vídeo: {self.video_path}")
            sys_logger.error("Nenhuma câmera física detectada!")
            raise RuntimeError("Não foi possível conectar a uma câmera física.")

        if not isinstance(source_to_open, str):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.running = True

        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        sys_logger.info("Fonte de vídeo inicializada com sucesso.")

    def _reopen_video(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.source)

    def _read_loop(self):
        next_frame_due = time.perf_counter()
        while self.running:
            if self.cap is None:
                time.sleep(0.01)
                continue

            if self.is_video_source and self.frame_interval > 0:
                delay = next_frame_due - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)

            ret, frame = self.cap.read()
            if ret and frame is not None:
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                with self.lock:
                    self.frame = frame
                    self.frame_seq += 1
                if self.is_video_source and self.frame_interval > 0:
                    next_frame_due += self.frame_interval
                    # Não acelera para “alcançar” o relógio: isso parece rewind/fast-forward.
                    now = time.perf_counter()
                    if next_frame_due < now - self.frame_interval:
                        next_frame_due = now
            else:
                if self.is_video_source:
                    self._reopen_video()
                    next_frame_due = time.perf_counter()
                else:
                    time.sleep(0.01)

    def peek_seq(self):
        with self.lock:
            return self.frame_seq

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None, 0
            return self.frame.copy(), self.frame_seq

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        sys_logger.info("Câmera encerrada.")
