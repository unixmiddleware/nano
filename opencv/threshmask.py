import cv2
print(cv2.__version__)
dispW=640
dispH=480
halfW=320
halfH=240
flip=2
framerate=21
fname = 'videos/myCam.avi'
win = 'nanoCam'
cvLogo = cv2.imread('cv.jpg')
cvLogo = cv2.resize(cvLogo,(halfW,halfH))
cvLogoGray = cv2.cvtColor(cvLogo,cv2.COLOR_BGR2GRAY)
cv2.imshow('cv Logo Gray',cvLogoGray)
cv2.moveWindow('cv Logo Gray', 0,halfH+130)

_,bgMask = cv2.threshold(cvLogoGray,225,255,cv2.THRESH_BINARY)
cv2.imshow('bgMask',bgMask)
cv2.moveWindow('bgMask',385,100)

fgMask = cv2.bitwise_not(bgMask)
cv2.imshow('FG Mask',fgMask)
cv2.moveWindow('FG Mask',385,halfH+130)

FG=cv2.bitwise_and(cvLogo,cvLogo,mask=fgMask)
cv2.imshow('FG',FG)
cv2.moveWindow('FG',703,350)

camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate='+str(framerate) +'/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cam = cv2.VideoCapture(camSet)
while True:
    ret, frame = cam.read()
    if ret:
        frame = cv2.resize(frame,(halfW,halfH))
        BG = cv2.bitwise_and(frame,frame,mask=bgMask)
        cv2.imshow('BG',BG)
        cv2.moveWindow('BG',703,100)
        cv2.imshow(win,frame)
        cv2.moveWindow(win,0,100)
        compImage=cv2.add(BG,FG)
        cv2.imshow('compImage',compImage)
        cv2.moveWindow('compImage',1017,100)

    if cv2.waitKey(1)==ord('q'): break
cam.release()
cv2.destroyAllWindows()

