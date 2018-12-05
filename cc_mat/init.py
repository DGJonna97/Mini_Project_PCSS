import cv2
import os
import numpy as np

from BLOB import BLOB
from BLOB import getBlobs

path = "C:/Evaluation"

#Function for segmenting the image. Following simlifyed method by Jonathan in watershed
def segment(image):
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

    return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3),np.uint8), iterations=1)

def withinRange(var, low, high):
    if(var < high and var > low):
        return True
    else:
        return False

def labelBlobs(imagePath):
    image = cv2.imread(imagePath, 0)

    seg = segment(image)

    #Getting connectedComponents from image
    _, components = cv2.connectedComponents(seg, connectivity=4)

    #Getting blobs from connectedComponents
    BLOBS = getBlobs(components)

    #Converting image to color since I want to draw colored boxed around blobs
    final = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    #For every blob I evaluate if it looks like a human (Currently only Area)
    #And draws a rectangle around it if it is.
    for x in range(len(BLOBS)):
        blob = BLOBS[x]

        if(withinRange(blob.getCompactness(), 0.4298, 0.6614)):

            x, y, w, h = blob.getRect()
            xCom, yCom = blob.getCenterOfMass()

            xCom = int((xCom * w) + x)
            yCom = int((yCom * h) + y)

            #Drawing bounding box
            cv2.rectangle(final, (x, y), (w+x, h+y), (0,255,0), 1)
            #Drawing Center Of Mass
            cv2.rectangle(final, (xCom-1, yCom-1), (xCom+1, yCom+1), (255,0,0), 1)

    return final

#Getting the base directory for the project
basedir = os.getcwd()
#specifying which folder the image files are in and getting an array of files
files = os.listdir(basedir + "/cc_mat/testset")

#for evey file in the folder /cc_mat/trainingset try to load the image and find blobs
for x in files:
    image = labelBlobs(os.path.abspath("cc_mat/testset/" + x))

    cv2.imwrite(os.path.join(path, x), image)
