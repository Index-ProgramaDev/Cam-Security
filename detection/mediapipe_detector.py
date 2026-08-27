"""
MediaPipeDetector — detecção de pose por crop individual de pessoa.

Pipeline esperado:
  YOLO → bounding box por pessoa → crop_person() → process_for_track() → landmarks no frame

IMPORTANTE — RunningMode.VIDEO:
  O detector usa RunningMode.VIDEO com detect_for_video() e timestamps monotônicos.
  Isso ativa o tracking interno do MediaPipe entre frames, estabilizando os landmarks
  e reduzindo o flickering. Cada track_id tem seu próprio detector para manter
  sequências temporais independentes por pessoa.

Parâmetros:
  - MIN_CROP_SIZE: 80px. Crops menores têm altíssima taxa de falha.
  - min_pose_detection_confidence: 0.45
  - min_pose_presence_confidence: 0.45
  - min_tracking_confidence: 0.45
  - num_poses: 1 — sempre 1 por crop (uma pessoa por crop).
  - output_segmentation_masks: False — não necessário, reduz overhead.

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

# Contador global de debug (desativar em produção setando para 0)
_DEBUG_INTERVAL = 90  # loga diagnóstico a cada N frames por track; 0 = desativado


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

    O crop pode ter padding arbitrário em relação à bbox YOLO original.
    crop_box = [x1, y1, x2, y2] — região do frame que foi enviada ao MediaPipe.

    Conversão:
        px = x1 + landmark.x * crop_width
        py = y1 + landmark.y * crop_height
        frame_norm_x = px / frame_width
        frame_norm_y = py / frame_height

    Não há resize/letterbox antes do MediaPipe, então a relação é linear direta.
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
        # Posição absoluta em pixels dentro do frame original
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


def _build_landmarker_options(model_path):
    """Constrói PoseLandmarkerOptions para RunningMode.VIDEO."""
    base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
    return mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.45,
        min_pose_presence_confidence=0.45,
        min_tracking_confidence=0.45,
        output_segmentation_masks=False,
    )


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
        self._last: dict = {}  # track_id -> (landmarks, wall_time)

    def update(self, track_id: int, landmarks: list):
        now = time.time()
        if landmarks:
            self._last[track_id] = (landmarks, now)
            return landmarks

        prev = self._last.get(track_id)
        if prev and (now - prev[1]) <= self.ttl_seconds:
            return prev[0]  # retorna última pose válida dentro do TTL

        return None

    def prune(self, active_ids: set):
        """Remove tracks inativos para evitar vazamento de memória."""
        stale = [tid for tid in self._last if tid not in active_ids]
        for tid in stale:
            self._last.pop(tid, None)


class _TrackLandmarker:
    """
    Detector MediaPipe Pose dedicado a um único track_id.

    Mantém timestamps monotônicos independentes por track para que o
    RunningMode.VIDEO funcione corretamente mesmo quando tracks têm
    frequências diferentes.
    """

    def __init__(self, model_path: str):
        options = _build_landmarker_options(model_path)
        self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        # Timestamp em ms enviado ao modelo; deve ser monotônico por instância.
        self._ts_ms: int = 0
        self._frame_counter: int = 0

    def detect(self, rgb_crop) -> list:
        """
        Executa detect_for_video com timestamp monotônico.
        Retorna lista de Landmark em coordenadas normalizadas do crop, ou [].
        """
        # Incrementa o timestamp em ~33 ms por chamada (≈ 30 FPS virtual)
        # Garante que o MediaPipe receba uma sequência temporal crescente,
        # mesmo que o pipeline real rode em frequências variáveis.
        self._ts_ms += 33
        self._frame_counter += 1

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
        result = self.landmarker.detect_for_video(mp_img, self._ts_ms)

        if not result.pose_landmarks:
            return []

        landmarks = _copy_landmarks(result.pose_landmarks[0])

        # Debug diagnóstico (a cada _DEBUG_INTERVAL frames, se ativado)
        if _DEBUG_INTERVAL > 0 and (self._frame_counter % _DEBUG_INTERVAL) == 1:
            n = len(landmarks)
            lm0 = landmarks[0] if n > 0 else None
            lm11 = landmarks[11] if n > 10 else None
            lm12 = landmarks[12] if n > 11 else None
            lm23 = landmarks[23] if n > 22 else None
            sys_logger.debug(
                f"[MediaPipe|diag] ts={self._ts_ms}ms landmarks={n} "
                f"lm0=({lm0.x:.3f},{lm0.y:.3f}) "
                f"lm11=({lm11.x:.3f},{lm11.y:.3f}) "
                f"lm12=({lm12.x:.3f},{lm12.y:.3f}) "
                f"lm23=({lm23.x:.3f},{lm23.y:.3f})"
                if lm0 and lm11 and lm12 and lm23 else
                f"[MediaPipe|diag] ts={self._ts_ms}ms landmarks={n}"
            )

        return landmarks

    def close(self):
        try:
            self.landmarker.close()
        except Exception:
            pass


class MediaPipeDetector:
    """
    Gerencia um pool de _TrackLandmarker — um por track_id ativo.

    Cada track tem seu próprio detector com sequência de timestamps independente,
    o que é obrigatório para RunningMode.VIDEO funcionar corretamente quando
    múltiplas pessoas são processadas em paralelo (ou em ordens diferentes
    a cada frame).
    """

    def __init__(self):
        self._model_path: str = ""
        self._pool: dict = {}   # track_id -> _TrackLandmarker
        self.ready = False
        self.pose_hold = PoseHold(ttl_seconds=POSE_HOLD_TTL)
        self._init_model_path()

    def _init_model_path(self):
        for candidate in ("pose_landmarker_lite.task", "pose_landmarker_full.task"):
            if os.path.exists(candidate):
                self._model_path = candidate
                self.ready = True
                sys_logger.info(
                    f"[MediaPipe] Modelo encontrado: {candidate} "
                    f"(RunningMode.VIDEO, 1 pose/crop)"
                )
                return
        sys_logger.warning("[MediaPipe] Nenhum modelo .task encontrado. Pose em standby.")

    def _get_landmarker(self, track_id: int) -> "_TrackLandmarker | None":
        """Retorna (criando se necessário) o detector para este track_id."""
        if not self.ready:
            return None
        if track_id not in self._pool:
            try:
                self._pool[track_id] = _TrackLandmarker(self._model_path)
                sys_logger.debug(f"[MediaPipe] Detector criado para track #{track_id}")
            except Exception as e:
                sys_logger.error(f"[MediaPipe] Falha ao criar detector para track #{track_id}: {e}")
                return None
        return self._pool[track_id]

    # ------------------------------------------------------------------
    # API principal — usar no pipeline YOLO→crop
    # ------------------------------------------------------------------

    def process_crop(self, track_id: int, crop) -> list:
        """
        Detecta uma pose no crop BGR da pessoa.

        1. Valida tamanho mínimo do crop.
        2. Converte BGR→RGB (MediaPipe exige RGB).
        3. Garante array C-contiguous.
        4. Chama detect_for_video com timestamp monotônico do track.
        5. Retorna lista de Landmark normalizados no espaço do crop (0..1).
        Retorna [] se falhar ou crop for muito pequeno.
        """
        if crop is None or crop.size == 0:
            return []

        h, w = crop.shape[:2]
        if w < MIN_CROP_SIZE or h < MIN_CROP_SIZE:
            return []

        landmarker = self._get_landmarker(track_id)
        if landmarker is None:
            return []

        try:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            if not rgb.flags["C_CONTIGUOUS"]:
                rgb = rgb.copy()
            return landmarker.detect(rgb)
        except Exception as e:
            sys_logger.error(f"[MediaPipe] Erro ao processar crop track #{track_id}: {e}")
            return []

    def process_for_track(self, track_id: int, crop, crop_box, frame_w: int, frame_h: int):
        """
        Processa o crop de uma pessoa e retorna os landmarks mapeados
        para o espaço do frame completo, com hold temporal anti-flickering.

        Parameters
        ----------
        track_id  : identificador único da pessoa (usado para pool e hold)
        crop      : imagem BGR recortada da pessoa (com padding)
        crop_box  : [x1, y1, x2, y2] no frame original — região enviada ao MediaPipe
        frame_w/h : dimensões do frame original

        Returns
        -------
        list[Landmark] em coordenadas normalizadas do frame (0..1), ou None se falhar.

        Nota sobre a conversão:
          Os landmarks retornados pelo MediaPipe estão normalizados em relação ao
          crop enviado. A conversão map_crop_landmarks_to_frame() reconstrói as
          coordenadas absolutas usando crop_box, sem qualquer heurística.
          Não há resize/letterbox antes do MediaPipe, então a relação é linear direta.
        """
        crop_lms = self.process_crop(track_id, crop)
        if crop_lms:
            mapped = map_crop_landmarks_to_frame(crop_lms, crop_box, frame_w, frame_h)
        else:
            mapped = []
        return self.pose_hold.update(track_id, mapped)

    # ------------------------------------------------------------------
    # Gerenciamento do pool
    # ------------------------------------------------------------------

    def prune_pool(self, active_ids: set):
        """
        Remove detectores de tracks que não estão mais ativos.
        Deve ser chamado a cada frame com o conjunto de track_ids visíveis.
        """
        stale = [tid for tid in self._pool if tid not in active_ids]
        for tid in stale:
            try:
                self._pool[tid].close()
            except Exception:
                pass
            del self._pool[tid]
            sys_logger.debug(f"[MediaPipe] Detector removido para track #{tid}")

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
        lms = self.process_crop(track_id=0, crop=frame)
        return {"pose_landmarks": [lms] if lms else []}

    def close(self):
        """Fecha todos os detectores do pool."""
        for tid, ld in list(self._pool.items()):
            try:
                ld.close()
            except Exception:
                pass
        self._pool.clear()
        self.ready = False
