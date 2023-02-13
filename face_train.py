import cv2 as cv
import numpy as np
import os
haar_cascade=cv.CascadeClassifier('haar_face.xml')
people=[]
for i in os.listdir(r'C:\Users\Lenovo\OneDrive\Desktop\Machine learning\OpenCV\opencv-course-master\Resources\Faces\train'):
    people.append(i)
DIR=r'C:\Users\Lenovo\OneDrive\Desktop\Machine learning\OpenCV\opencv-course-master\Resources\Faces\train'


features=[]
labels=[]

def create_train():
    for person in people:
        path=os.path.join(DIR,person)
        label=people.index(person)
        for img in os.listdir(path):
            img_path=os.path.join(path,img)
            img_array=cv.imread(img_path)
            gray=cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)
            


            faces_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)
            for(x,y,w,h) in faces_rect:
                faces_roi=gray[y:y+h,x:x+w]
                features.append(faces_roi)
                labels.append(label)
create_train()
print('training done................................................................')

print(f'the lenght of the features = {len(features)}')
print(f'the lenght of the labels = {len(labels)}')

face_recognizer=cv.face.LBPHFaceRecognizer_create()
features,labels=np.array(features,dtype='object'),np.array(labels)
face_recognizer.train(features,labels)
face_recognizer.save('face_trained.yml')
np.save('features.npy',features)
np.save('labels.npy',labels)