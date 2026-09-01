import os
import time
import subprocess
import shutil
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np

from utils.logger import sys_logger

STORAGE_ROOT       = "storage"
EVENTS_DIR         = os.path.join(STORAGE_ROOT, "events")
SNAPSHOTS_DIR      = os.path.join(STORAGE_ROOT, "snapshots")
PERIODIC_SNAP_DIR  = os.path.join(STORAGE_ROOT, "periodic_snapshots")
TEMP_DIR           = os.path.join(STORAGE_ROOT, "temp")
EVIDENCES_DIR      = os.path.join(STORAGE_ROOT, "evidences")

FFMPEG_CRF    = 23
FFMPEG_PRESET = "fast"

STORAGE_MAX_AGE_DAYS     = 7
STORAGE_MAX_SIZE_MB      = 500
CLEANUP_CHECK_INTERVAL   = 300.0
PERIODIC_FRAME_THRESHOLD = 40
PERIODIC_TIME_THRESHOLD  = 2.0

_FFMPEG_FALLBACK_PATHS = [
    r"C:\Users\Usuario\Downloads\evolution-api\node_modules\@ffmpeg-installer\win32-x64\ffmpeg.exe",
    r"C:\ffmpeg\ffmpeg.exe",
    r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
    r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
]


@dataclass
class EvidenceMeta:
    event_id:      str
    storage_path:  str
    snapshot_path: str
    mime_type:     str           = "video/mp4"
    size_bytes:    int           = 0
    duration_ms:   int           = 0
    created_at:    float         = 0.0
    ok:            bool          = True
    error:         Optional[str] = None
    track_id:      Optional[int] = None
    face_id:       Optional[str] = None
    person_id:     Optional[str] = None


def _find_ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    for path in _FFMPEG_FALLBACK_PATHS:
        if os.path.isfile(path):
            sys_logger.info(f"[EvidenceCapture] ffmpeg encontrado em: {path}")
            return path
    return ""


def _ffmpeg_available() -> bool:
    return bool(_find_ffmpeg())


class EvidenceCapture:
    def __init__(self, fps: float = 30.0):
        self.fps = fps
        for d in (EVENTS_DIR, SNAPSHOTS_DIR, TEMP_DIR, EVIDENCES_DIR):
            os.makedirs(d, exist_ok=True)

    def build_evidence(self, pre_frames, post_frames, event_id, triggered_at,
                       track_id=None, face_id=None, person_id=None) -> EvidenceMeta:
        all_frames = pre_frames + post_frames
        if not all_frames:
            return EvidenceMeta(event_id=event_id, storage_path="", snapshot_path="",
                                ok=False, error="Nenhum frame disponível", created_at=time.time())

        ds       = time.localtime(triggered_at)
        event_dir = os.path.join(EVENTS_DIR, time.strftime("%Y", ds),
                                 time.strftime("%m", ds), time.strftime("%d", ds))
        os.makedirs(event_dir, exist_ok=True)

        output_path   = os.path.join(event_dir, f"event_{event_id}.mp4")
        snapshot_path = os.path.join(SNAPSHOTS_DIR, f"event_{event_id}_trigger.jpg")
        tmp_dir       = os.path.join(TEMP_DIR, f"evt_{event_id}")
        os.makedirs(tmp_dir, exist_ok=True)

        try:
            fps = self._estimate_fps(all_frames)
            self._save_snapshot(all_frames, triggered_at, snapshot_path)
            n = self._write_frames(all_frames, tmp_dir)
            if n == 0:
                raise RuntimeError("Nenhum frame gravado.")

            duration_ms = int((n / fps) * 1000)
            sys_logger.info(f"[EvidenceCapture] Gerando vídeo: {n} frames @ {fps:.1f}fps ≈ {duration_ms/1000:.1f}s → {output_path}")

            if not self._run_ffmpeg(tmp_dir, output_path, fps):
                raise RuntimeError("FFmpeg falhou.")

            size_bytes = os.path.getsize(output_path) if os.path.exists(output_path) else 0
            sys_logger.info(f"[EvidenceCapture] Vídeo gerado: {output_path} ({size_bytes//1024}KB, {duration_ms}ms)")

            return EvidenceMeta(
                event_id=event_id,
                storage_path=os.path.abspath(output_path),
                snapshot_path=os.path.abspath(snapshot_path) if os.path.exists(snapshot_path) else "",
                size_bytes=size_bytes, duration_ms=duration_ms,
                created_at=time.time(), ok=True,
                track_id=track_id, face_id=face_id, person_id=person_id,
            )
        except Exception as e:
            sys_logger.error(f"[EvidenceCapture] Erro ao gerar evidência {event_id}: {e}")
            return EvidenceMeta(event_id=event_id, storage_path="", snapshot_path="",
                                ok=False, error=str(e), created_at=time.time(),
                                track_id=track_id, face_id=face_id, person_id=person_id)
        finally:
            self._cleanup(tmp_dir)

    def _estimate_fps(self, frames) -> float:
        if len(frames) < 2:
            return self.fps
        d = frames[-1][0] - frames[0][0]
        return max(1.0, min(120.0, (len(frames) - 1) / d)) if d > 0 else self.fps

    def _save_snapshot(self, frames, triggered_at: float, path: str):
        if not frames:
            return
        closest = min(frames, key=lambda x: abs(x[0] - triggered_at))
        try:
            cv2.imwrite(path, closest[1], [cv2.IMWRITE_JPEG_QUALITY, 90])
        except Exception as e:
            sys_logger.warning(f"[EvidenceCapture] Falha ao salvar snapshot: {e}")

    def _write_frames(self, frames, tmp_dir: str) -> int:
        written = 0
        for idx, (_, frame) in enumerate(frames):
            if frame is None:
                continue
            try:
                cv2.imwrite(os.path.join(tmp_dir, f"frame_{idx+1:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                written += 1
            except Exception as e:
                sys_logger.warning(f"[EvidenceCapture] Falha ao escrever frame {idx+1}: {e}")
        return written

    def _run_ffmpeg(self, frame_dir: str, output_path: str, fps: float) -> bool:
        ffmpeg = _find_ffmpeg()
        if not ffmpeg:
            sys_logger.error("[EvidenceCapture] ffmpeg não encontrado. Instale em https://ffmpeg.org/download.html")
            return False
        cmd = [ffmpeg, "-y", "-framerate", str(fps), "-i", os.path.join(frame_dir, "frame_%06d.jpg"),
               "-c:v", "libx264", "-crf", str(FFMPEG_CRF), "-preset", FFMPEG_PRESET,
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", output_path]
        try:
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
            if r.returncode != 0:
                sys_logger.error(f"[EvidenceCapture] ffmpeg falhou:\n{r.stderr.decode('utf-8', errors='replace')[:1000]}")
                return False
            return True
        except subprocess.TimeoutExpired:
            sys_logger.error("[EvidenceCapture] ffmpeg timeout após 120s.")
            return False
        except Exception as e:
            sys_logger.error(f"[EvidenceCapture] Erro ao executar ffmpeg: {e}")
            return False

    def _cleanup(self, tmp_dir: str):
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception as e:
            sys_logger.warning(f"[EvidenceCapture] Falha ao limpar temp {tmp_dir}: {e}")


class StorageCleaner:
    def __init__(self,
                 max_age_days: float = STORAGE_MAX_AGE_DAYS,
                 max_size_mb: float = STORAGE_MAX_SIZE_MB,
                 check_interval: float = CLEANUP_CHECK_INTERVAL):
        self.max_age_days   = max_age_days
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.check_interval = check_interval
        self._last_check: float = 0.0
        self._managed_dirs = [EVENTS_DIR, SNAPSHOTS_DIR, PERIODIC_SNAP_DIR, TEMP_DIR, EVIDENCES_DIR]

    def run_if_due(self) -> bool:
        now = time.time()
        if (now - self._last_check) < self.check_interval:
            return False
        self._last_check = now
        self.run_cleanup()
        return True

    def run_cleanup(self):
        removed_count = 0
        removed_bytes = 0
        cutoff_time   = time.time() - (self.max_age_days * 86400)

        all_files: List[Tuple[float, int, str]] = []
        for d in self._managed_dirs:
            if not os.path.isdir(d):
                continue
            for root, _, files in os.walk(d):
                for fn in files:
                    fpath = os.path.join(root, fn)
                    try:
                        all_files.append((os.path.getmtime(fpath), os.path.getsize(fpath), fpath))
                    except OSError:
                        pass

        all_files.sort(key=lambda x: x[0])
        current_size = sum(f[1] for f in all_files)

        for mtime, size, fpath in all_files:
            if mtime < cutoff_time or current_size > self.max_size_bytes:
                try:
                    os.remove(fpath)
                    removed_count += 1
                    removed_bytes += size
                    current_size  -= size
                except OSError as e:
                    sys_logger.warning(f"[StorageCleaner] Falha ao remover {fpath}: {e}")

        self._cleanup_empty_dirs()

        if removed_count > 0:
            sys_logger.info(
                f"[StorageCleaner] {removed_count} arquivo(s) removidos "
                f"({removed_bytes // (1024*1024)}MB). Atual: ~{current_size // (1024*1024)}MB"
            )

    def _cleanup_empty_dirs(self):
        for d in self._managed_dirs:
            if not os.path.isdir(d):
                continue
            for root, dirs, _ in os.walk(d, topdown=False):
                for subdir in dirs:
                    subpath = os.path.join(root, subdir)
                    try:
                        if not os.listdir(subpath):
                            os.rmdir(subpath)
                    except OSError:
                        pass


class PeriodicSnapshotter:
    def __init__(self,
                 frame_threshold: int = PERIODIC_FRAME_THRESHOLD,
                 time_threshold_seconds: float = PERIODIC_TIME_THRESHOLD):
        self.frame_threshold = frame_threshold
        self.time_threshold  = time_threshold_seconds
        self._frame_count    = 0
        self._last_capture_ts: float = 0.0
        self._seq = 0
        os.makedirs(PERIODIC_SNAP_DIR, exist_ok=True)

    def on_frame(self, frame, timestamp: Optional[float] = None) -> Optional[str]:
        if frame is None:
            return None
        self._frame_count += 1
        now = timestamp or time.time()

        if self._frame_count < self.frame_threshold and (now - self._last_capture_ts) < self.time_threshold:
            return None

        return self._capture(frame, now)

    def force_capture(self, frame, timestamp: Optional[float] = None) -> Optional[str]:
        if frame is None:
            return None
        return self._capture(frame, timestamp or time.time())

    def _capture(self, frame, ts: float) -> Optional[str]:
        try:
            ds = time.localtime(ts)
            subdir = os.path.join(PERIODIC_SNAP_DIR,
                                  time.strftime("%Y", ds),
                                  time.strftime("%m", ds),
                                  time.strftime("%d", ds))
            os.makedirs(subdir, exist_ok=True)

            self._seq += 1
            fpath = os.path.join(subdir, f"snap_{time.strftime('%H%M%S', ds)}_{self._seq:06d}.jpg")
            cv2.imwrite(fpath, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

            self._frame_count     = 0
            self._last_capture_ts = ts
            return os.path.abspath(fpath)
        except Exception as e:
            sys_logger.warning(f"[PeriodicSnapshotter] Falha: {e}")
            return None

    def reset(self):
        self._frame_count     = 0
        self._last_capture_ts = 0.0
