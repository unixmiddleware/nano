import cv2
import numpy as np
print(cv2.__version__)

def click(event,x,y,flags,params):
    global evt
    global pnt
    evt = event
    if event==cv2.EVENT_LBUTTONDOWN:
        pnt = (x,y)
        if not pnt in coords: coords.append(pnt)
    if event==cv2.EVENT_RBUTTONDOWN:
        blue = int(frame[y,x,0])
        green = int(frame[y,x,1])
        red = int(frame[y,x,2])

        img[:] = [(blue,green,red)] 
        tp =(255-blue,255-green,255-red)
        colorString='{},{},{}'.format(blue,green,red)
        cv2.putText(img,colorString,(10,25),fnt,1,tp,2)
        cv2.imshow('imgWin',img)

pnt = None
evt = None
coords = []
img = np.zeros((250,250,3),np.uint8)

dispW=640
dispH=480
flip=2
framerate=21
win = 'nanoCam'
cv2.namedWindow(win)
cv2.setMouseCallback(win,click)
fnt = cv2.FONT_HERSHEY_PLAIN

camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate='+str(framerate) +'/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cam = cv2.VideoCapture(camSet)
while True:
    ret, frame = cam.read()
    if ret:
        frame = cv2.resize(frame,(dispW,dispH))
        for coord in coords:
            loc = str(coord)
            cv2.putText(frame,loc,coord,fnt,1,(255,0,0),2)
            cv2.circle(frame,coord,5,(0,0,255),-1)

        cv2.imshow(win,frame)
        cv2.moveWindow(win,0,0)
    keyPressed = cv2.waitKey(1)
    if keyPressed==ord('q'): break
    if keyPressed==ord('c'): coords=[]
cam.release()
cv2.destroyAllWindows()