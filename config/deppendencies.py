def get_yolo():
    from ultralytics import YOLO;
    modelo = YOLO("yolov8n.pt")
    return modelo