import threading
from typing import Optional, Dict

from utils.logger import sys_logger
from evidence.video_buffer import VideoBuffer
from evidence.evidence_capture import EvidenceCapture
from evidence.evidence_store import save_evidence_meta


class EvidenceManager:
    def __init__(self, fps: float = 30.0):
        self.fps      = fps
        self._buffers: Dict[str, VideoBuffer] = {}
        self._capture = EvidenceCapture(fps=fps)
        self._lock    = threading.Lock()

    def register_camera(self, camera_id: str, fps: Optional[float] = None):
        with self._lock:
            if camera_id not in self._buffers:
                self._buffers[camera_id] = VideoBuffer(camera_id=camera_id, fps=fps or self.fps)
                sys_logger.info(f"[EvidenceManager] Buffer registrado para câmera '{camera_id}'")

    def push_frame(self, camera_id: str, frame, timestamp: Optional[float] = None):
        buf = self._get_buffer(camera_id)
        if buf:
            buf.push(frame, timestamp)

    def on_event(self, event_id: str, camera_id: str = "cam_0",
                 track_id: Optional[int] = None, face_id: Optional[str] = None,
                 person_id: Optional[str] = None):
        buf = self._get_buffer(camera_id)
        if not buf:
            sys_logger.warning(f"[EvidenceManager] Câmera '{camera_id}' não registrada — evidência ignorada.")
            return

        def _on_ready(pre_frames, post_frames, ev_id, triggered_at):
            sys_logger.info(f"[EvidenceManager] Frames prontos para event {ev_id}: pre={len(pre_frames)} post={len(post_frames)}")
            meta = self._capture.build_evidence(
                pre_frames=pre_frames, post_frames=post_frames,
                event_id=ev_id, triggered_at=triggered_at,
                track_id=track_id, face_id=face_id, person_id=person_id,
            )
            if not meta.ok:
                sys_logger.error(f"[EvidenceManager] Falha na geração de evidência event={ev_id}: {meta.error}")
                return
            eid = save_evidence_meta(
                event_id=ev_id, storage_path=meta.storage_path, mime_type=meta.mime_type,
                size_bytes=meta.size_bytes, duration_ms=meta.duration_ms, evidence_type="video",
                track_id=track_id, face_id=face_id, person_id=person_id,
            )
            sys_logger.info(f"[EvidenceManager] Evidência registrada: evidence_id={eid} event={ev_id} track={track_id} face={face_id} person={person_id} → {meta.storage_path}")
            if meta.snapshot_path:
                save_evidence_meta(event_id=ev_id, storage_path=meta.snapshot_path,
                                   mime_type="image/jpeg", evidence_type="snapshot",
                                   track_id=track_id, face_id=face_id, person_id=person_id)

        buf.trigger(event_id=event_id, on_ready_callback=_on_ready)

    def _get_buffer(self, camera_id: str) -> Optional[VideoBuffer]:
        with self._lock:
            return self._buffers.get(camera_id)
