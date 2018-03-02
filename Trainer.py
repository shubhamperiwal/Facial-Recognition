#library to capture all the paths
import os
import cv2
import numpy as np
#need pillow to capture images. PIL is Python Image Library
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
#relative path of the samples
path = 'dataset'

def getImagesWithId(path):
     #Concatenates root path with image name
     #from dataset folder, list all directories (pictures in the folder) and appending it to path with the separator
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    IDs = []
    for imagePath in imagePaths:        
        #if it is not, then it'll convert image to grayscale
        faceImg = Image.open(imagePath).convert('L')
        #open the image, convert to numpy array
        #OpenCV only works with NUMPY array
        faceNp = np.array(faceImg, 'uint8')
        #extract ID from 'dataset\\User1.1.jpg'
        ID = int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow("Training", faceNp)
        cv2.waitKey(10)
    return np.array(IDs), faces


IDs, faces = getImagesWithId(path)

#now we train the recognizer
recognizer.train(faces, IDs)
recognizer.save('recognizer/trainingData.yml')
cv2.destroyAllWindows()