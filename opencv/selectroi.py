import cv2
print(cv2.__version__)
coords = []

def click(event,x,y,flags,params):
    pnt = (x,y)
    if event==cv2.EVENT_LBUTTONDOWN:
        if len(coords) == 0: coords.append(pnt)
    if event==cv2.EVENT_LBUTTONUP:
        if len(coords) == 1: coords.append(pnt)
    return event

dispW=640
dispH=480
flip=2
framerate=21
win = 'nanoCam'
fnt = cv2.FONT_HERSHEY_PLAIN
cv2.namedWindow(win) 
cv2.setMouseCallback(win,click)

camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate='+str(framerate) +'/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cam = cv2.VideoCapture(camSet)
while True:
    ret, frame = cam.read()
    if ret:
        frame = cv2.resize(frame,(dispW,dispH))
#        print('addr,global',addr,id(coords))
        if len(coords) > 1:
            loc1 = str(coords[0])
            loc2 = str(coords[1])
            x1 = coords[0][0]
            x2 = coords[1][0]
            y1 = coords[0][1]
            y2 = coords[1][1]
            
            cv2.putText(frame,loc1,coords[0],fnt,1,(255,0,0),2)
            cv2.putText(frame,loc2,coords[1],fnt,1,(255,0,0),2)
            roi = frame[y1:y2,x1:x2]
            cv2.imshow('ROI',roi)
            cv2.moveWindow('ROI', dispW+70, 0)
            cv2.rectangle(frame,coords[0],coords[1],(0,0,255),2)

        cv2.imshow(win,frame)
        cv2.moveWindow(win,0,0)
    keyPressed = cv2.waitKey(1)
    if keyPressed==ord('q'): break
    if keyPressed==ord('c'): coords=[]
cam.release()
cv2.destroyAllWindows()