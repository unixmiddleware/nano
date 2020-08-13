import cv2
print(cv2.__version__)
dispW=640
dispH=480
flip=2
framerate=21
fname = 'videos/myCam.avi'

camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate='+str(framerate) +'/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
#cam = cv2.VideoCapture(camSet)
cam = cv2.VideoCapture(fname)
#outVid = cv2.VideoWriter(fname,cv2.VideoWriter_fourcc(*'XVID'),framerate,(dispW,dispH))
while True:
    ret, frame = cam.read()
    if frame is not None:
        sizedImage = cv2.resize(frame,(dispW,dispH))
        cv2.imshow('nanoCam',sizedImage)
        cv2.moveWindow('nanoCam',0,0)
    #   outVid.write(frame)
    if cv2.waitKey(50)==ord('q'):
        break
cam.release()
#outVid.release()
cv2.destroyAllWindows()

