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

cv.createTrackbar('hue2Lo','trackbars',hueLo,maxHue,nothing)
cv.createTrackbar('hue2Hi','trackbars',hueHi,maxHue,nothing)

camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate='+str(framerate) +'/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cam = cv.VideoCapture(camSet)
png = cv.imread('opencv/smarties.png')

while True:
    ret,frame = cam.read()
    if not frame is None:
        hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
        hueLo = cv.getTrackbarPos('hueLo','trackbars')
        hueHi = cv.getTrackbarPos('hueHi','trackbars')
        hue2Lo = cv.getTrackbarPos('hue2Lo','trackbars')
        hue2Hi = cv.getTrackbarPos('hue2Hi','trackbars')
        satLo = cv.getTrackbarPos('satLo','trackbars')
        satHi = cv.getTrackbarPos('satHi','trackbars')
        valLo = cv.getTrackbarPos('valLo','trackbars')
        valHi = cv.getTrackbarPos('valHi','trackbars')
        
        loBound = (hueLo,satLo,valLo)
        hiBound = (hueHi,satHi,valHi)
        loBound2 = (hue2Lo,satLo,valLo)
        hiBound2 = (hue2Hi,satHi,valHi)

        FGmask=cv.inRange(hsv,loBound,hiBound)
        FGmask2=cv.inRange(hsv,loBound2,hiBound2)
        FGmaskComp=cv.add(FGmask,FGmask2)

        cv.imshow('FGmaskComp',FGmaskComp) 
        cv.moveWindow('FGmaskComp',0,300)

        FG = cv.bitwise_and(frame,frame,mask=FGmaskComp)
        cv.imshow('FG',FG)
        cv.moveWindow('FG',390,0)

        BGmask = cv.bitwise_not(FGmaskComp)
        cv.imshow('BGmask',BGmask)
        cv.moveWindow('BGmask',390,300)

        BG = cv.cvtColor(BGmask,cv.COLOR_GRAY2BGR)
        cv.imshow('BG',BG)
        cv.moveWindow('BG',720,300)

        final = cv.add(FG,BG)
        cv.imshow('final',final)
        cv.moveWindow('final',720,0)

        cv.imshow(win,frame)
        cv.moveWindow(win,0,0)
    if cv.waitKey(30)==ord('q'): break
cam.release()
cv.destroyAllWindows()

