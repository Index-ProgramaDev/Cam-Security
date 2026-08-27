"""Benchmark headless do pipeline YOLO → track → pose/crop → face intervalada."""
import time
import cv2
from tracking.id_manager import IDManager
from tracking.object_tracker import ObjectTracker
from detection.person_detector import PersonDetector, crop_person
from detection.face_detector import FaceDetector
from detection.mediapipe_detector import MediaPipeDetector
from face_biometry.face_capture import FaceCapture, FACE_CHECK_INTERVAL
from face_biometry.face_storage import FaceStorage
from face_biometry.face_reid import FaceReID

# Intervalo de re-verificação quando a identidade já foi estabelecida (pode ser maior)
FACE_RECHECK_IDENTIFIED = 2.0


def should_check_face(info, now):
    last = info.get("last_face_check") or 0.0
    identified = info.get("identity") is not None
    interval = FACE_RECHECK_IDENTIFIED if identified else FACE_CHECK_INTERVAL
    return (now - last) >= interval


def run(video_path, max_frames=90, width=640, height=480):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Não abriu {video_path}")

    person_detector = PersonDetector()
    pose_detector = MediaPipeDetector()
    face_detector = FaceDetector()
    face_capture = FaceCapture(face_detector=face_detector)
    face_storage = FaceStorage()
    face_reid = FaceReID(face_storage)
    tracker = ObjectTracker(IDManager(), ttl_seconds=300.0)

    face_calls = 0
    pose_ok = 0
    people_hist = []
    t0 = time.perf_counter()
    n = 0

    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            if not ok:
                break
        frame = cv2.resize(frame, (width, height))
        now = time.time()
        h, w = frame.shape[:2]
        dets = person_detector.detect_persons(frame)
        tracks = tracker.update(dets, frame=frame)
        pose_detector.pose_hold.prune(set(tracks.keys()))
        people_hist.append(len(tracks))
        for tid, info in tracks.items():
            crop, crop_box = crop_person(frame, info["box"], pad=True)
            pose = pose_detector.process_for_track(tid, crop, crop_box, w, h) if crop.size else None
            if pose:
                pose_ok += 1
            if should_check_face(info, now) and crop.size:
                face_calls += 1
                insights = face_capture.capture_face_insights(crop)
                info["last_face_check"] = now
                if insights and insights.get("embedding") is not None:
                    face_storage.save_embedding(tid, insights["embedding"])
                    matched = face_reid.match_embedding(insights["embedding"])
                    if matched is not None and matched != tid:
                        info["identity"] = matched
                mem = tracker.tracks.get(tid)
                if mem is not None:
                    mem["last_face_check"] = info["last_face_check"]
                    mem["identity"] = info.get("identity")
        n += 1

    elapsed = time.perf_counter() - t0
    cap.release()
    pose_detector.close()
    face_detector.close()
    fps = n / elapsed if elapsed else 0
    return {
        "frames": n,
        "seconds": round(elapsed, 2),
        "fps": round(fps, 2),
        "avg_people": round(sum(people_hist) / max(len(people_hist), 1), 2),
        "max_people": max(people_hist) if people_hist else 0,
        "pose_ok_tracks": pose_ok,
        "face_calls": face_calls,
        "face_calls_per_frame": round(face_calls / max(n, 1), 2),
    }


if __name__ == "__main__":
    for path in ("config/luta.mp4", "config/video2.mp4", "config/video.mp4"):
        print("===", path, "===")
        try:
            print(run(path, max_frames=60))
        except Exception as e:
            print("FALHA:", e)
