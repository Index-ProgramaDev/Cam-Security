"""
MediaPipeDetector — detecção de pose por crop individual de pessoa.

Pipeline esperado:
  YOLO → bounding box por pessoa → crop_person() → process_for_track() → landmarks no frame

Parâmetros de confiança ajustados para 640×480:
  - MIN_CROP_SIZE: 80px (antes 64). Crops menores têm altíssima taxa de falha.
  - min_pose_detection_confidence: 0.45 (antes 0.35) — menos falsos positivos instáveis.
  - min_pose_presence_confidence: 0.45 (antes 0.40) — descarta poses parciais.
  - min_tracking_confidence: 0.45 (antes 0.50) — ligeiramente mais tolerante para
    manter continuidade entre frames sem degradar a qualidade.
  - num_poses: 1 — sempre 1 por crop (uma pessoa por crop).

PoseHold:
  - Mantém a última pose válida por POSE_HOLD_TTL segundos quando o MediaPipe
    falha temporariamente (flickering). Não cria poses falsas persistentes.
  - Após o TTL expirar, a pose é descartada normalmente.
"""

import cv2
import time
import os
import mediapipe as mp
from utils.logger import sys_logger

# Tamanho mínimo do crop para tentar detecção (pixels)
MIN_CROP_SIZE = 80

# Tempo máximo de retenção de uma pose válida quando o detector falha (segundos)
POSE_HOLD_TTL = 0.35

# Visibilidade mínima para desenhar um landmark
MIN_LANDMARK_VISIBILITY = 0.30


class Landmark:
    __slots__ = ("x", "y", "z", "visibility", "presence")

    def __init__(self, x, y, z=0.0, visibility=1.0, presence=1.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.visibility = float(visibility)
        self.presence = float(presence)


def _copy_landmarks(raw_landmarks):
    """Copia landmarks do resultado MediaPipe para objetos simples e serializáveis."""
    copied = []
    for lm in raw_landmarks:
        copied.append(
            Landmark(
                lm.x,
                lm.y,
                getattr(lm, "z", 0.0),
                getattr(lm, "visibility", 1.0),
                getattr(lm, "presence", 1.0),
            )
        )
    return copied


def map_crop_landmarks_to_frame(landmarks, crop_box, frame_w, frame_h):
    """
    Converte landmarks normalizados do crop (0..1 relativo ao crop)
    para coordenadas normalizadas do frame completo (0..1 relativo ao frame).

    Essa conversão é crítica: sem ela, o skeleton aparece deslocado.
    """
    if not landmarks:
        return []

    x1, y1, x2, y2 = crop_box
    crop_w = max(1, x2 - x1)
    crop_h = max(1, y2 - y1)
    fw = max(frame_w, 1)
    fh = max(frame_h, 1)

    mapped = []
    for lm in landmarks:
        # Posição absoluta em pixels dentro do frame
        px = x1 + lm.x * crop_w
        py = y1 + lm.y * crop_h
        mapped.append(
            Landmark(
                px / fw,
                py / fh,
                lm.z,
                lm.visibility,
                lm.presence,
            )
        )
    return mapped


class PoseHold:
    """
    Mantém a última pose válida por um curto período (POSE_HOLD_TTL) para suavizar
    frames onde o MediaPipe falha temporariamente.

    Comportamento:
      - Se landmarks válidos chegam → atualiza cache e retorna.
      - Se landmarks vazios chegam E o cache não expirou → retorna cache.
      - Se landmarks vazios chegam E o cache expirou → retorna None.
    """

    def __init__(self, ttl_seconds: float = POSE_HOLD_TTL):
        self.ttl_seconds = ttl_seconds
        self._last: dict = {}  # track_id -> (landmarks, timestamp)

    def update(self, track_id: int, landmarks: list):
        now = time.time()
        if landmarks:
            self._last[track_id] = (landmarks, now)
            return landmarks

        prev = self._last.get(track_id)
        if prev and (now - prev[1]) <= self.ttl_seconds:
            return prev[0]  # retorna última pose válida

        return None

    def prune(self, active_ids: set):
        """Remove tracks inativos para evitar vazamento de memória."""
        stale = [tid for tid in self._last if tid not in active_ids]
        for tid in stale:
            self._last.pop(tid, None)


class MediaPipeDetector:
    def __init__(self):
        self.detector = None
        self.ready = False
        self.pose_hold = PoseHold(ttl_seconds=POSE_HOLD_TTL)
        self._init_detector()

    def _init_detector(self):
        try:
            # Prefere o modelo lite; cai no full se não encontrar
            for candidate in ("pose_landmarker_lite.task", "pose_landmarker_full.task"):
                if os.path.exists(candidate):
                    model_path = candidate
                    break
            else:
                sys_logger.warning(
                    "[MediaPipe] Nenhum modelo .task encontrado. Pose em standby."
                )
                return

            base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
            options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_poses=1,                         # sempre 1 por crop
                min_pose_detection_confidence=0.45,  # aumentado de 0.35
                min_pose_presence_confidence=0.45,   # aumentado de 0.40
                min_tracking_confidence=0.45,        # ligeiramente reduzido de 0.50
            )
            self.detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)
            self.ready = True
            sys_logger.info(
                f"[MediaPipe] PoseLandmarker (1 pose/crop, IMAGE) carregado: {model_path}"
            )
        except Exception as e:
            sys_logger.error(f"[MediaPipe] Erro ao carregar PoseDetector: {e}")
            self.ready = False

    # ------------------------------------------------------------------
    # API principal — usar no pipeline YOLO→crop
    # ------------------------------------------------------------------

    def process_crop(self, crop):
        """
        Detecta uma pose no crop da pessoa.
        Retorna lista de Landmark normalizados no espaço do crop (0..1).
        Retorna [] se falhar ou crop for muito pequeno.
        """
        if not self.ready or crop is None or crop.size == 0:
            return []

        h, w = crop.shape[:2]
        if w < MIN_CROP_SIZE or h < MIN_CROP_SIZE:
            return []

        try:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            if not rgb.flags["C_CONTIGUOUS"]:
                rgb = rgb.copy()
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = self.detector.detect(mp_img)
            if not res.pose_landmarks:
                return []
            return _copy_landmarks(res.pose_landmarks[0])
        except Exception as e:
            sys_logger.error(f"[MediaPipe] Erro ao processar crop: {e}")
            return []

    def process_for_track(self, track_id: int, crop, crop_box, frame_w: int, frame_h: int):
        """
        Processa o crop de uma pessoa e retorna os landmarks mapeados
        para o espaço do frame completo, com hold temporal anti-flickering.

        Parameters
        ----------
        track_id  : identificador único da pessoa
        crop      : imagem BGR recortada da pessoa (com padding)
        crop_box  : [x1, y1, x2, y2] no frame original (usado para conversão)
        frame_w/h : dimensões do frame original

        Returns
        -------
        list[Landmark] em coordenadas normalizadas do frame, ou None se falhar.
        """
        crop_lms = self.process_crop(crop)
        if crop_lms:
            mapped = map_crop_landmarks_to_frame(crop_lms, crop_box, frame_w, frame_h)
        else:
            mapped = []
        return self.pose_hold.update(track_id, mapped)

    # ------------------------------------------------------------------
    # Compatibilidade — não usar no pipeline novo
    # ------------------------------------------------------------------

    def process(self, frame):
        """
        Mantido apenas para compatibilidade com código legado.
        No pipeline YOLO→crop NÃO deve ser chamado — use process_for_track().
        """
        if frame is None:
            return {"pose_landmarks": []}
        lms = self.process_crop(frame)
        return {"pose_landmarks": [lms] if lms else []}

    def close(self):
        if self.ready and self.detector:
            self.detector.close()
            self.ready = False
