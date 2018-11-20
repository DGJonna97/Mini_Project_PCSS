import cv2
import numpy as np

from BLOB import BLOB
from BLOB import getBlobs

#Function for segmenting the image. Following simlifyed method by Jonathan in watershed
def segment(image):
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

    return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3),np.uint8), iterations=1)

image = cv2.imread('img2.png', 0)

image = segment(image)

#Getting connectedComponents from image
_, components = cv2.connectedComponents(image, connectivity=4)

#Getting blobs from connectedComponents
BLOBS = getBlobs(components)

#Converting image to color since I want to draw colored boxed around blobs
image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

#For every blob I evaluate if it looks like a human (Currently only Area)
#And draws a rectangle around it if it is.
for x in range(len(BLOBS)):
    blob = BLOBS[x]

    if(blob.getArea() != 0):

        x, y, w, h = blob.getRect()
        xCom, yCom = blob.getCenterOfMass()

        print(blob.getPerimeter())
        print(blob.getCircularity())

        #Drawing bounding box
        cv2.rectangle(image, (x, y), (w+x, h+y), (0,255,0), 1)
        #Drawing Center Of Mass
        cv2.rectangle(image, (xCom-1, yCom-1), (xCom+1, yCom+1), (255,0,0), 1)

cv2.imshow("final", image)
cv2.waitKey(0)
