import cv2
print(cv2.__version__)
dispW=640
dispH=480
x1=140
y1=100
x2=180
y2=140
xstep=10
ystep=10
flip=2
framerate=21
fnt = cv2.FONT_HERSHEY_DUPLEX

camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate='+str(framerate) +'/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cam = cv2.VideoCapture(camSet)
while True:
    ret, frame = cam.read()
    if ret:
        if x2 + xstep > dispW: xstep = -xstep
        if y2 + ystep > dispH: ystep = -ystep
        if y1 + ystep < 0: ystep = -ystep
        if x1 + xstep < 0: xstep = -xstep
        x1 += xstep
        x2 += xstep
        y1 += ystep
        y2 += ystep
        msg = 'x1,y1={},{} x2,y2={},{}'.format(x1,y1,x2,y2)
        frame=cv2.resize(frame,(dispW,dispH))
        frame=cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),-1)
        frame=cv2.putText(frame,msg,(100,300),fnt,1,(255,0,150),2)
        cv2.imshow('nanoCam',frame)
        cv2.moveWindow('nanoCam',0,0)
    if cv2.waitKey(20)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()

