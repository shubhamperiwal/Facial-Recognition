import cv2
import numpy as np
import sql_connector

def authorise():
    #Boolean to decide wether authorised face or not
    isAuthorised = False

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
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)
        for(x,y,w,h) in faces:
            cv2.rectangle (img, (x,y), (x+w, y+h), (0,0,255), 2)
            ID, conf = rec.predict(gray[y:y+h, x:x+w])
            #str(id) is the text you want to print next to the face. #(x, y+h) is location of text
            name = sql_connector.retrieve(ID)
            cv2.putText(img, name, (x,y+h), fontface, fontscale, fontcolor)         

            #Authorise if authorised person
            if name == "Shubham":
                isAuthorised = True
                
        cv2.imshow("Face", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
   
    #Disconnect
    cam.release()
    cv2.destroyAllWindows()    

    return isAuthorised