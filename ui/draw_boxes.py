import cv2
from ultralytics import YOLO;

resultados = YOLO("yolov8n.pt")

_, contornos = cv2.findContours()

def desenhar_caixas(x, y, w, h):
    for r in resultados:
        cv2.rectangle(frame, 50, 50, 5  (255, 0, 0), 2)