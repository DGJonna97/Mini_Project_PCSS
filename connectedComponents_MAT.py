import numpy as np
import cv2 as cv

img = cv.imread('img3.png', 1)

Bparams = cv.SimpleBlobDetector_Params()

Bparams.minDistBetweenBlobs = 0

Bparams.filterByArea = True
Bparams.minArea = 20

Bparams.filterByCircularity = False
Bparams.filterByConvexity = False
Bparams.filterByInertia = False


Btector = cv.SimpleBlobDetector_create(Bparams)

#cv.imshow("Original", img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Binary_inv sets the minimum and max value. The OTSU is an algorithm to choose the best threshold.
ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_TRIANGLE)

#cv.imshow("Black and white", thresh)

# Remove noise
kernel = np.ones((3, 3),np.uint8)
# Erosion of the image.
opening = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=1)

blobs = Btector.detect(opening)

_, labels = cv.connectedComponents(opening)

# Map component labels to hue val
label_hue = np.uint8(179*labels/np.max(labels))
blank_ch = 255*np.ones_like(label_hue)
labeled_img = cv.merge([label_hue, blank_ch, blank_ch])

# cvt to BGR for display
labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)

# set bg label to black
labeled_img[label_hue==0] = 0

print(len(labels))

kpts_img = cv.drawKeypoints(img, blobs, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.imshow("binary img", opening)
cv.imshow("Keypoints", labeled_img)


cv.waitKey(0)
cv.destroyAllWindows()
