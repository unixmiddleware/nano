import cv2
print(cv2.__version__)
dispW=640
dispH=480
flip=2
framerate=21

win = 'nanoCam'
face_cascade=cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
eyes_cascade=cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate='+str(framerate) +'/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cam = cv2.VideoCapture(camSet)
while True:
    ret, frame = cam.read()
    if ret:
        frame = cv2.resize(frame,(dispW,dispH))
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        faces = face_cascade.detectMultiScale(gray,1.3,5)
        eyes = eyes_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
        for (x,y,w,h) in eyes:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
        cv2.imshow('Gray',gray)
        cv2.moveWindow('Gray',720,0)
        cv2.imshow(win,frame)
        cv2.moveWindow(win,0,0)
    if cv2.waitKey(1)==ord('q'): break
cam.release()
cv2.destroyAllWindows()

