import Dataset_Creator
import Trainer
import cv2
import sql_connector
import numpy as np

#Create facedetect object
faceDetect = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
cam=cv2.VideoCapture(0)

#create recognizer
rec = cv2.face.LBPHFaceRecognizer_create()

#load training data so we have a trained recognizer
rec.read("recognizer/trainingData.yml")
ID=0

#create font and color of text
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (255, 255, 255)

while(True):
    #ret is a return value, true if the frame is returned correctly
    #img is the frame it reads
    ret, img = cam.read()

    #since openCV takes in only grayscale, we have to convert it
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h) in faces:

        #Draws the rectangle next to the place
        cv2.rectangle (img, (x,y), (x+w, y+h), (0,0,255), 2)

        #Recognises the rectangle
        ID, conf = rec.predict(gray[y:y+h, x:x+w])

        #Retrieves name from database 
        name = sql_connector.retrieve(ID)

        #str(id) is the text you want to print next to the face. (x, y+h) is location of text
        cv2.putText(img, name, (x,y+h), fontface, fontscale, fontcolor)         
    
    #display the resulting frame
    cv2.imshow("Face", img)

    #Wait key is the speed at which the frames are displayed to us
    if(cv2.waitKey(1)==ord('q')):
        break

#release camera
cam.release()
cv2.destroyAllWindows()





