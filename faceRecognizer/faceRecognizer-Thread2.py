import face_recognition
import cv2
import os 
import pickle
import time

class Camera:
    def __init__(self,cameraType,window):
        self.window = window
        self.cameraType = cameraType
        self.frameStart = time.time()
        self.fpsReport = 0
        if cameraType == 'webcam':
            self.dispW=640
            self.dispH=400
            self.xpos = 0
            self.ypos = 0
            self.cam = cv2.VideoCapture('/dev/video1')
        elif cameraType == 'picam':
            self.dispW=640
            self.dispH=400
            self.xpos = 0
            self.ypos = self.dispH + 10
            self.flip=2
            self.framerate=21
            self.camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate='+str(self.framerate) +'/1 ! nvvidconv flip-method='+str(self.flip)+' ! video/x-raw, width='+str(self.dispW)+', height='+str(self.dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
            self.cam = cv2.VideoCapture(self.camSet)
    def stop(self):
        self.cam.release()
    def FrameRate(self,frame):
        dx = time.time() - self.frameStart
        if dx > 0:
            fps = 1/dx
            if self.fpsReport == 0: self.fpsReport = fps
            self.fpsReport = (.95 * self.fpsReport) + (.05 * fps)
            fpsMessage = self.cameraType + ' ' + str(round(self.fpsReport,2)) + 'fps'
            bottom = self.dispH // 10
            right = self.dispW // 3
            cv2.rectangle(frame,(0,0),(right,bottom),(255,0,255),-1)
            cv2.putText(frame,fpsMessage,(0,20),font,.75,(0,255,255))
    def recognise(self):
        scaleFactor = 0.25
        self.frameStart = time.time()
        _,frame = self.cam.read()
        frameSmall = cv2.resize(frame,(0,0),fx=scaleFactor,fy=scaleFactor)
        frameRGB   = cv2.cvtColor(frameSmall,cv2.COLOR_BGR2RGB)
        facePositions = face_recognition.face_locations(frameRGB,model='cnn')
        allEncodings  = face_recognition.face_encodings(frameRGB,facePositions)
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
        self.FrameRate(frame)
        cv2.imshow(self.window,frame)
        cv2.moveWindow(self.window,self.xpos,self.ypos)


print(cv2.__version__)

trained_data = os.path.join('traindata','face-trained.pkl')
with open(trained_data,'rb') as fr:
    Names = pickle.load(fr)
    Encodings = pickle.load(fr)

cam1 = Camera('webcam','webcam')
cam2 = Camera('picam','picam')

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    cam1.recognise()
    cam2.recognise()

    if cv2.waitKey(1) == ord('q'):
        break
cam1.stop()
cam2.stop()
cv2.destroyAllWindows()

