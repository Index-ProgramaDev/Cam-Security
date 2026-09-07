import cv2
import time
from ultralytics import YOLO

def main():
    print("Iniciando Benchmark YOLO...")
    model = YOLO("yolov8n.pt")
    
    resolutions = [640, 512, 416, 320]
    
    for res in resolutions:
        cap = cv2.VideoCapture("config/daniel.mp4")
        times = []
        confs = []
        boxes_count = []
        
        for i in range(20):
            ret, frame = cap.read()
            if not ret: break
            
            t0 = time.perf_counter()
            results = model(frame, imgsz=res, verbose=False, conf=0.50)[0]
            t1 = time.perf_counter()
            
            times.append((t1 - t0) * 1000)
            
            frame_boxes = 0
            for det in results.boxes:
                if int(det.cls[0]) == 0:
                    frame_boxes += 1
                    confs.append(float(det.conf[0]))
            boxes_count.append(frame_boxes)
            
        cap.release()
        
        avg_time = sum(times) / len(times) if times else 0
        avg_conf = sum(confs) / len(confs) if confs else 0
        avg_boxes = sum(boxes_count) / len(boxes_count) if boxes_count else 0
        
        print(f"Resolução: {res}x{res} | Latência: {avg_time:.1f}ms | Conf Média: {avg_conf:.2f} | Pessoas Média: {avg_boxes:.1f}")

if __name__ == "__main__":
    main()
