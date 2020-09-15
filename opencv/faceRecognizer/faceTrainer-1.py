import face_recognition
import glob
import os 
import pickle

traindata = os.path.join('traindata','face-trained.pkl')
known_dir = os.path.join('demoImages','known')

known_jpgs = glob.glob(os.path.join(known_dir,'*.jpg'))
known_images = [face_recognition.load_image_file(x) for x in known_jpgs]
Encodings = [face_recognition.face_encodings(x)[0] for x in known_images]
Names = [os.path.basename(x).replace('.jpg','') for x in known_jpgs]

with open(traindata,'wb') as fw:
    pickle.dump(Names,fw)
    pickle.dump(Encodings,fw)

