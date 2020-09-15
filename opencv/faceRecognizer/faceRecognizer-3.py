import face_recognition
import cv2
import glob
import os 
import pickle

print(cv2.__version__)
unknown_dir = os.path.join('demoImages','unknown')

trained_data = os.path.join('traindata','face-trained.pkl')
with open(trained_data,'rb') as fr:
    Names = pickle.load(fr)
    Encodings = pickle.load(fr)

font = cv2.FONT_HERSHEY_SIMPLEX
unknown_jpgs = glob.glob(os.path.join(unknown_dir,'*.jpg'))
for unknown_jpg in unknown_jpgs:
    testImage     = face_recognition.load_image_file(unknown_jpg)
    facePositions = face_recognition.face_locations(testImage)
    allEncodings  = face_recognition.face_encodings(testImage,facePositions)

    testImage = cv2.cvtColor(testImage,cv2.COLOR_RGB2BGR)
    for (top,right,bottom,left), face_encoding in zip(facePositions,allEncodings):
        name = 'Unknown Person'
        matches = face_recognition.compare_faces(Encodings,face_encoding)
        if True in matches:
            match_index = matches.index(True)
            name = Names[match_index]
        cv2.rectangle(testImage,(left,top),(right,bottom),(0,0,255),2)
        cv2.putText(testImage,name,(left,top-6),font,0.75,(0,255,255),2)
    cv2.imshow('myWindow',testImage)
    cv2.moveWindow('myWindow',0,0)
    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()
