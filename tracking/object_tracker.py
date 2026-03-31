import cv2

from config.deppendencies import get_yolo

yolo = get_yolo()


cap = cv2.VideoCapture(0)


while True:
 
    ret, frame = cap.read()

    results = yolo.track(frame, persist=True)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO26 Tracking", annotated_frame)

      
    if cv2.waitKey(0) == 27:
            break


cap.release()
cv2.destroyAllWindows()