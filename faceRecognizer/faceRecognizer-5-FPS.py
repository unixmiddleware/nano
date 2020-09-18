import face_recognition
import cv2
import os 
import pickle
import time

fpsReport = 0
scaleFactor = 0.25
print(cv2.__version__)

trained_data = os.path.join('traindata','face-trained.pkl')
with open(trained_data,'rb') as fr:
    Names = pickle.load(fr)
    Encodings = pickle.load(fr)

cam = cv2.VideoCapture('/dev/video1')
font = cv2.FONT_HERSHEY_SIMPLEX

frameStart = time.time()
while True:
    frameStart = time.time()
    _,frame = cam.read()
    frameSmall = cv2.resize(frame,(0,0),fx=scaleFactor,fy=scaleFactor)
    frameRGB = cv2.cvtColor(frameSmall,cv2.COLOR_BGR2RGB)
    facePositions = face_recognition.face_locations(frameRGB,model='cnn')
    allEncodings = face_recognition.face_encodings(frameRGB,facePositions)
    for (top,right,bottom,left), face_encoding in zip(facePositions,allEncodings):
        name = 'Unknown Person'
        matches = face_recognition.compare_faces(Encodings,face_encoding)
        if True in matches:
            match_index = matches.index(True)
            name = Names[match_index]
            top    = int(top    / scaleFactor)
            bottom = int(bottom / scaleFactor)
            left   = int(left   / scaleFactor)
            right  = int(right  / scaleFactor)
            cv2.rectangle(frame,(left,top),(right,bottom),(0,0,255),2)
            cv2.putText(frame,name,(left,top-6),font,0.75,(0,0,255))
    frameEnd = time.time()
    dx = frameEnd - frameStart
    if dx > 0:
        fps = 1/dx
        if fpsReport == 0: fpsReport = fps
        fpsReport = (.95 * fpsReport) + (.05 * fps)
        fpsMessage = str(round(fpsReport,2)) + 'fps'
        cv2.rectangle(frame,(0,0),(100,40),(0,0,255),-1)
        cv2.putText(frame,fpsMessage,(0,20),font,.75,(0,255,255))
        print(fpsMessage)
    cv2.imshow('Picture',frame)
    cv2.moveWindow('Picture',0,0)

    if cv2.waitKey(1) == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()

