import face_recognition
import cv2
import os 
import pickle
import time
from threading import Thread
import numpy as np

class Camera:
    def __init__(self,cameraType,window,src):
        self.window = window
        self.cameraType = cameraType
        self.frameStart = time.time()
        self.fpsReport = 0
        self.dispW=640
        self.dispH=400
        self.bottom = self.dispH // 10
        self.right = self.dispW // 3
        self.xpos = 0
        self.ypos = 0
        if cameraType == 'webcam':
            self.cam = cv2.VideoCapture(src)
        elif cameraType == 'picam':
            self.ypos = self.dispH + 10
            self.flip=2
            self.framerate=21
            self.camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate='+str(self.framerate) +'/1 ! nvvidconv flip-method='+str(self.flip)+' ! video/x-raw, width='+str(self.dispW)+', height='+str(self.dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
            self.cam = cv2.VideoCapture(self.camSet)
        self.thread = Thread(target=self.update)
        self.thread.daemon = True
        self.thread.start()
        print('Init ',cameraType,window,src)
    def stop(self):
        self.cam.release()
    def FrameMessage(self,frame,start,end):
        self.fpsReport, self.fpsMessage = FrameRate(start,end,self.fpsReport)
        cv2.rectangle(frame,(0,0),(self.right,self.bottom),(255,0,255),-1)
        cv2.putText(frame,self.fpsMessage,(0,20),font,.75,(0,255,255))
    def update(self):
        while True:
            self.frameStart = time.time()
            _,self.frame = self.cam.read()
            if not self.frame is None:
                self.frameEnd = time.time()
                self.frame = cv2.resize(self.frame,(self.dispW,self.dispH))
    def recognise(self):
        faces = ExamineFrame(self.frame)

        for face in faces:
            cv2.rectangle(self.frame,(face['left'],face['top']),(face['right'],face['bottom']),(0,0,255),2)
            cv2.putText(self.frame,face['name'],(face['left'],face['top']-6),font,0.75,(0,0,255))
        self.FrameMessage(self.frame,self.frameStart,self.frameEnd)
        return self.frame

def ExamineFrame(frame):
    faces = []

    frameSmall = cv2.resize(frame,(0,0),fx=scaleFactor,fy=scaleFactor)
    frameRGB   = cv2.cvtColor(frameSmall,cv2.COLOR_BGR2RGB)
    facePositions = face_recognition.face_locations(frameRGB,model='cnn')
    allEncodings  = face_recognition.face_encodings(frameRGB,facePositions)
    for (top,right,bottom,left),face_encoding in zip(facePositions,allEncodings):
        name = 'Unknown Person'
        matches = face_recognition.compare_faces(Encodings,face_encoding)
        if True in matches:
            match_index = matches.index(True)
            name = Names[match_index]
            top    = int(top    / scaleFactor)
            bottom = int(bottom / scaleFactor)
            left   = int(left   / scaleFactor)
            right  = int(right  / scaleFactor)
            detected = {'top': top, 'bottom': bottom, 'left': left, 'right': right, 'name': name}
            faces.append(detected)
    return faces

def FrameRate(frameStart,frameEnd,fpsReport):
    dx = frameEnd - frameStart
    if dx > 0:
        fps = 1/dx
        if fpsReport == 0: fpsReport = fps
        fpsReport = (.95 * fpsReport) + (.05 * fps)
        fpsMessage = str(round(fpsReport,2)) + 'fps'
    return fpsReport,fpsMessage

print(cv2.__version__)

trained_data = os.path.join('traindata','face-trained.pkl')
with open(trained_data,'rb') as fr:
    Names = pickle.load(fr)
    Encodings = pickle.load(fr)

scaleFactor = 0.25
cam1 = Camera('webcam','webcam1','/dev/video1')
cam2 = Camera('webcam','webcam2','/dev/video2')
cam3 = Camera('picam','picam','')

font = cv2.FONT_HERSHEY_SIMPLEX

frame1 = frame2 = frame3 = None
while True:
    try:
        x = cam1.recognise()
        if not x is None: frame1 = x
    except:
        pass
    try:
        x = cam2.recognise()
        if not x is None: frame2 = x
    except:
        pass
    try:
        x = cam3.recognise()
        if not x is None: frame3 = x
    except:
        pass

    if not frame1 is None:
        cv2.imshow('webcam1',frame1)
        cv2.moveWindow('webcam1',0,0)
    if not frame2 is None:
        cv2.imshow('webcam2',frame2)
        cv2.moveWindow('webcam2',800,0)
    if not frame3 is None:
        cv2.imshow('picam',frame3)
        cv2.moveWindow('picam',0,600)

    if cv2.waitKey(1) == ord('q'):
        exit(1)
        break

cam1.stop()
cam2.stop()
cam3.stop()
cv2.destroyAllWindows()