import cv2
import time
from utils.logger import sys_logger
from detection.person_detector import PersonDetector, crop_person
from detection.mediapipe_detector import MediaPipeDetector

def main():
    sys_logger.info("Iniciando Profiling do Pose...")
    cap = cv2.VideoCapture("config/daniel.mp4")
    if not cap.isOpened():
        print("Erro ao abrir config/daniel.mp4")
        return

    person_detector = PersonDetector()
    pose_detector = MediaPipeDetector()

    frame_count = 0
    max_frames = 20

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        persons = person_detector.detect_persons(frame)
        if persons:
            for p in persons:
                crop, crop_box = crop_person(frame, p["box"], pad=True)
                if crop.size > 0:
                    pose_detector.process_for_track(1, crop, crop_box, frame.shape[1], frame.shape[0], frame_ts_ms=frame_count*33)
        frame_count += 1
    
    cap.release()
    pose_detector.close()
    sys_logger.info("Profiling concluído.")

if __name__ == "__main__":
    main()
