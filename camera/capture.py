import cv2

cap = cv2.VideoCapture(0)

def IniciarCap():
    

 while True:
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    movimento = cv2.absdiff(frame1 - frame2)
    blur = cv2.GaussianBlur(movimento, (5, 5), 0)
    
    frame1 = frame2
    frame2 = frame1
    
    cv2.imshow("Camera", blur)
    
if cv2.waitKey(0) == 27:
    cap.release()
    cv2.destroyAllWindows()