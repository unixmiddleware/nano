import face_recognition
import cv2
import os 
import pickle

print(cv2.__version__)

trained_data = os.path.join('traindata','face-trained.pkl')
with open(trained_data,'rb') as fr:
    Names = pickle.load(fr)
    Encodings = pickle.load(fr)

cam = cv2.VideoCapture('/dev/video1')
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    _,frame = cam.read()
    print('New Frame')
    frameSmall = cv2.resize(frame,(0,0),fx=.25,fy=.25)
    frameRGB = cv2.cvtColor(frameSmall,cv2.COLOR_BGR2RGB)
    facePositions = face_recognition.face_locations(frameRGB,model='cnn')
    allEncodings = face_recognition.face_encodings(frameRGB,facePositions)
    for (top,right,bottom,left), face_encoding in zip(facePositions,allEncodings):
        name = 'Unknown Person'
        matches = face_recognition.compare_faces(Encodings,face_encoding)
        if True in matches:
            match_index = matches.index(True)
            name = Names[match_index]
            top *= 4
            bottom *= 4
            left *= 4
            right *= 4
            cv2.rectangle(frame,(left,top),(right,bottom),(0,0,255),2)
            cv2.putText(frame,name,(left,top-6),font,0.75,(0,0,255))
    cv2.imshow('Picture',frame)
    cv2.moveWindow('Picture',0,0)
    if cv2.waitKey(1) == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()

