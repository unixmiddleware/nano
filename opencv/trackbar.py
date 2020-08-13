import cv2
print(cv2.__version__)
dispW=640
dispH=480
flip=2
framerate=21
win = 'nanoCam'
xbar = 'xbar'
ybar = 'ybar'
wbar = 'wbar'
hbar = 'hbar'
def nothing(x):
    return
camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate='+str(framerate) +'/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cam = cv2.VideoCapture(camSet)
cv2.namedWindow(win)
cv2.createTrackbar(xbar,win,25,dispW,nothing)
cv2.createTrackbar(ybar,win,25,dispH,nothing)
cv2.createTrackbar(wbar,win,25,dispW,nothing)
cv2.createTrackbar(hbar,win,25,dispH,nothing)
while True:
    ret, frame = cam.read()
    
    if ret:
        frame = cv2.resize(frame,(dispW,dispH))
        xVal = cv2.getTrackbarPos(xbar,win)
        yVal = cv2.getTrackbarPos(ybar,win)
        wVal = cv2.getTrackbarPos(wbar,win)
        hVal = cv2.getTrackbarPos(hbar,win)
        frame=cv2.rectangle(frame,(xVal,yVal),(xVal+wVal,yVal+hVal),(255,0,0),3)
        cv2.imshow(win,frame)
        cv2.moveWindow(win,0,0)
    if cv2.waitKey(1)==ord('q'): break
cam.release()
cv2.destroyAllWindows()
