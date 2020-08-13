import cv2 as cv
import numpy as np 

print(cv.__version__)
dispW=320
dispH=240
flip=2
framerate=21
win = 'nanoCam'

maxHue = 360//2
maxSat = 255
maxVal = 255

hueLo = 50
hueHi = maxHue
satLo = 50
satHi = maxSat
valLo = 50
valHi = maxVal

def nothing(x):
    pass

cv.namedWindow('trackbars')
cv.moveWindow('trackbars',1320,0)

cv.createTrackbar('hueLo','trackbars',hueLo,maxHue,nothing)
cv.createTrackbar('hueHi','trackbars',hueHi,maxHue,nothing)
cv.createTrackbar('satLo','trackbars',satLo,maxSat,nothing)
cv.createTrackbar('satHi','trackbars',satHi,maxSat,nothing)
cv.createTrackbar('valLo','trackbars',valLo,maxVal,nothing)
cv.createTrackbar('valHi','trackbars',valHi,maxVal,nothing)

camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate='+str(framerate) +'/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cam = cv.VideoCapture(camSet)
png = cv.imread('opencv/smarties.png')

while True:
    ret,frame = cam.read()
    if not frame is None:
        hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
        hueLo = cv.getTrackbarPos('hueLo','trackbars')
        hueHi = cv.getTrackbarPos('hueHi','trackbars')
        satLo = cv.getTrackbarPos('satLo','trackbars')
        satHi = cv.getTrackbarPos('satHi','trackbars')
        valLo = cv.getTrackbarPos('valLo','trackbars')
        valHi = cv.getTrackbarPos('valHi','trackbars')
        
        loBound = (hueLo,satLo,valLo)
        hiBound = (hueHi,satHi,valHi)

        FGmask=cv.inRange(hsv,loBound,hiBound)
        cv.imshow('FGmask',FGmask) 
        cv.moveWindow('FGmask',0,410)

        FG = cv.bitwise_and(frame,frame,mask=FGmask)
        cv.imshow('FG',FG)
        cv.moveWindow('FG',480,0)

        BGmask = cv.bitwise_not(FGmask)
        cv.imshow('BGmask',BGmask)
        cv.moveWindow('BGmask',480,410)

        BG = cv.cvtColor(BGmask,cv.COLOR_GRAY2BGR)
        cv.imshow('BG',BG)
        cv.moveWindow('BG',900,410)

        final = cv.add(FG,BG)
        cv.imshow('final',final)
        cv.moveWindow('final',900,0)

        cv.imshow(win,frame)
        cv.moveWindow(win,0,0)
    if cv.waitKey(30)==ord('q'): break
cam.release()
cv.destroyAllWindows()

