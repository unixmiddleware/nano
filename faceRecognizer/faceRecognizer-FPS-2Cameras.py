import face_recognition
import cv2
import os 
import pickle
import time

fpsReport = 0
scaleFactor = 0.25
dispW=640
dispH=480
flip=2
framerate=21

print(cv2.__version__)

trained_data = os.path.join('traindata','face-trained.pkl')
with open(trained_data,'rb') as fr:
    Names = pickle.load(fr)
    Encodings = pickle.load(fr)

cam1 = cv2.VideoCapture('/dev/video1')

camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate='+str(framerate) +'/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cam2 = cv2.VideoCapture(camSet)
font = cv2.FONT_HERSHEY_SIMPLEX

frameStart = time.time()
while True:
    frameStart = time.time()
    _,frame1 = cam1.read()
    _,frame2 = cam2.read()

    frame1Small = cv2.resize(frame1,(0,0),fx=scaleFactor,fy=scaleFactor)
    frame1RGB = cv2.cvtColor(frame1Small,cv2.COLOR_BGR2RGB)
    frame2Small = cv2.resize(frame2,(0,0),fx=scaleFactor,fy=scaleFactor)
    frame2RGB = cv2.cvtColor(frame2Small,cv2.COLOR_BGR2RGB)
    
    face1Positions = face_recognition.face_locations(frame1RGB,model='cnn')
    allEncodings = face_recognition.face_encodings(frame1RGB,face1Positions)
    for (top,right,bottom,left), face_encoding in zip(face1Positions,allEncodings):
        name = 'Unknown Person'
        matches = face_recognition.compare_faces(Encodings,face_encoding)
        if True in matches:
            match_index = matches.index(True)
            name = Names[match_index]
            top    = int(top    / scaleFactor)
            bottom = int(bottom / scaleFactor)
            left   = int(left   / scaleFactor)
            right  = int(right  / scaleFactor)
            cv2.rectangle(frame1,(left,top),(right,bottom),(0,0,255),2)
            cv2.putText(frame1,name,(left,top-6),font,0.75,(0,0,255))

    face2Positions = face_recognition.face_locations(frame2RGB,model='cnn')
    allEncodings = face_recognition.face_encodings(frame2RGB,face1Positions)
    for (top,right,bottom,left), face_encoding in zip(face2Positions,allEncodings):
        name = 'Unknown Person'
        matches = face_recognition.compare_faces(Encodings,face_encoding)
        if True in matches:
            match_index = matches.index(True)
            name = Names[match_index]
            top    = int(top    / scaleFactor)
            bottom = int(bottom / scaleFactor)
            left   = int(left   / scaleFactor)
            right  = int(right  / scaleFactor)
            cv2.rectangle(frame2,(left,top),(right,bottom),(0,0,255),2)
            cv2.putText(frame2,name,(left,top-6),font,0.75,(0,0,255))
    frameEnd = time.time()
    dx = frameEnd - frameStart
    if dx > 0:
        fps = 1/dx
        if fpsReport == 0: fpsReport = fps
        fpsReport = (.95 * fpsReport) + (.05 * fps)
        fpsMessage = str(round(fpsReport,2)) + 'fps'
        cv2.rectangle(frame1,(0,0),(100,40),(0,0,255),-1)
        cv2.putText(frame1,fpsMessage,(0,20),font,.75,(0,255,255))
        print(fpsMessage)
    cv2.imshow('Picture1',frame1)
    cv2.moveWindow('Picture1',0,0)
    cv2.imshow('Picture2',frame2)
    cv2.moveWindow('Picture2',0,400)

    if cv2.waitKey(1) == ord('q'):
        break
cam1.release()
cv2.destroyAllWindows()

