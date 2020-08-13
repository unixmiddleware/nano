import cv2
print(cv2.__version__)
dispW=640
dispH=480
flip=2
framerate=21
fname = 'videos/myCam.avi'
win = 'nanoCam'

camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate='+str(framerate) +'/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cam = cv2.VideoCapture(camSet)
while True:
    ret, frame = cam.read()
    if ret:
        frame = cv2.resize(frame,(dispW,dispH))
        cv2.imshow(win,frame)
        cv2.moveWindow(win,0,0)
    if cv2.waitKey(1)==ord('q'): break
cam.release()
cv2.destroyAllWindows()

