import cv2
def capture(cam):
    while True:
        if cv2.waitKey(1)==ord('q'):
            break
        ret,frame = cam.read()
        cv2.imshow('piCam',frame)

print(cv2.__version__)
dispW=340
dispH=240
flip=2
camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cam=cv2.VideoCapture(camSet)
capture(cam)
cam.release()
cv2.destroyAllWindows()
