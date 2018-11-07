import cv2
import numpy as np

from BLOB import BLOB
from BLOB import getBlobs

def segment(image):
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

    return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3),np.uint8), iterations=1)

image = cv2.imread('img2.png', 0)

image = segment(image)

_, components = cv2.connectedComponents(image, connectivity=4)

BLOBS = getBlobs(components)

print(BLOBS[0].getArea())

cv2.imshow("Test", image)
cv2.waitKey(0)
