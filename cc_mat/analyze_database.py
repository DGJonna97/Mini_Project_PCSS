import cv2
import os
import numpy as np

from BLOB import BLOB
from BLOB import getBlobs

def segment(image):
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

    return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3),np.uint8), iterations=1)

basedir = os.getcwd()
files = os.listdir(basedir + "/testimg")

file = open(basedir + "/database_values.txt", "w")

for x in files:
    print(x + " -- " +os.path.abspath("testimg/" + x))
    image = cv2.imread(os.path.abspath("testimg/" + x), 0)

    file.write(x + " -:" + "\n")

    image = segment(image)

    _, components = cv2.connectedComponents(image, connectivity=4)

    cv2.imshow("Image", image)
    BLOBS = getBlobs(components)
    print(len(BLOBS))
    cv2.waitKey(0)

    if(len(BLOBS) != 1):
        file.write("    Error: Too many/few blobs (" + str(len(BLOBS)) + " BLOBs found)")
    else:
        file.write("    area: "         + str(BLOBS[0].getArea())         + "\n")
        file.write("    centerOfMass: " + str(BLOBS[0].getCenterOfMass()) + "\n")
        file.write("    rect: "         + str(BLOBS[0].getRect())         + "\n")
        file.write("    compactness: "  + str(BLOBS[0].getCompactness())  + "\n")
        file.write("    perimeter: "    + str(BLOBS[0].getPerimeter())    + "\n")
        file.write("    circularity: "  + str(BLOBS[0].getCircularity())  + "\n")

    file.write("\n")

file.close()
