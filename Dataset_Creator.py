import cv2
import numpy as np

#Create facedetect object
faceDetect = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
#Catpure video from webcam
cam=cv2.VideoCapture(0)

id=raw_input('Enter user id: ')
name=raw_input('Enter name: ')
sampleNumber=0
while(True):
    ret, img = cam.read()
    
    #Convert color image to gray scale for classifier to work
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #detects all faces in frame and returns coordinates
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h) in faces:
        sampleNumber+=1
        #Save face in a folder
        cv2.imwrite("dataset/User."+str(id)+"."+str(sampleNumber)+".jpg", gray[y:y+h, x:x+w])        
        cv2.rectangle (img, (x,y), (x+w, y+h), (0,0,255), 2)
        cv2.waitKey(100)
    cv2.imshow("Face", img)
    cv2.waitKey(1)
    if(sampleNumber>20):
        break

#release camera
cam.release()
cv2.destroyAllWindows()




