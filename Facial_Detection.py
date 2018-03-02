import cv2
import numpy as np

#Create facedetect object
faceDetect = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
#Catpure video from webcam
cam=cv2.VideoCapture(0)

while(True):
    ret, img = cam.read()
    
    #Convert color image to gray scale for classifier to work
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #detects all faces in frame and returns coordinates
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h) in faces:
        #Draw rectangle on face. initial is (x,y) end with x+w, y+h
        # 2 in the end is thickness
        cv2.rectangle (img, (x,y), (x+w, y+h), (0,0,255), 2)
    cv2.imshow("Face", img)
    if(cv2.waitKey(1)==ord('q')):
        break
#release camera
cam.release()
cv2.destroyAllWindows()




