import cv2
import os
import numpy as np

from BLOB import BLOB
from BLOB import getBlobs

def segment(image):

    image = cv2.medianBlur(image, 3)

    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

    return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3),np.uint8), iterations=1)

#Getting the base directory for the project
basedir = os.getcwd()
#specifying which folder the image files are in and getting an array of files
files = os.listdir(basedir + "/cc_mat/trainingsetV2")

#initiating file to write to
file = open(basedir + "/database_values.txt", "w")

#for evey file in the folder /cc_mat/trainingset try to load the image and find blobs
for x in files:
    print(x + " -- " +os.path.abspath("cc_mat/trainingsetV2/" + x))
    image = cv2.imread(os.path.abspath("cc_mat/trainingsetV2/" + x), 0)

    file.write(x + " -:" + "\n")

    image = segment(image)
    cv2.imshow("Pictures", image)
    cv2.waitKey(0)

    _, components = cv2.connectedComponents(image, connectivity=4)

    BLOBS = getBlobs(components)

    #If there are more or less then one BLOB write error to file
    if(len(BLOBS) != 1):
        file.write("    Error: Too many/few blobs (" + str(len(BLOBS)) + " BLOBs found)" + "\n")
    else:
    #If there is only one BLOB write it's featues to the file
        file.write("    area: "         + str(BLOBS[0].getArea())         + "\n")
        file.write("    centerOfMass: " + str(BLOBS[0].getCenterOfMass()) + "\n")
        file.write("    rect: "         + str(BLOBS[0].getRect())         + "\n")
        file.write("    compactness: "  + str(BLOBS[0].getCompactness())  + "\n")
        file.write("    perimeter: "    + str(BLOBS[0].getPerimeter())    + "\n")
        file.write("    circularity: "  + str(BLOBS[0].getCircularity())  + "\n")

    file.write("\n")

#Close and save the text file
file.close()
