from pyexpat import features
import cv2 as cv
import numpy as np

haar_cascade=cv.CascadeClassifier('haar_face.xml')
from face_train import people

# features=np.load('features.npy')
# labels=np.load('labels.npy')
face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img=cv.imread(r'C:\Users\Lenovo\OneDrive\Desktop\Machine learning\OpenCV\opencv-course-master\Resources\Faces\train\Madonna\17.jpg')
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow('person',gray)
faces_rect=haar_cascade.detectMultiScale(gray,1.1,4)


for (x,y,w,h) in faces_rect:
    face_roi=gray[y:y+h,x:x+w]

    label,confidence=face_recognizer.predict(face_roi)
    print(f'Label={people[label]} with a confidence of {confidence}')
    cv.putText(img,str(people[label]),(100,200),cv.FONT_HERSHEY_SIMPLEX,1.0,(255,0,0),thickness=2)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
cv.imshow('Detected Face',img)
cv.waitKey(0)