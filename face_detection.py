import cv2 as cv

img=cv.imread('grppic.jpg')
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
haar_cascade=cv.CascadeClassifier('haar_face.xml')
# cv.imshow('image',gray)


faces_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=7)


print(f"number of faces found in the image= {len(faces_rect)}")
for( x,y,w,h) in faces_rect:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)

cv.imshow('deteced faces',img)

cv.waitKey(0)