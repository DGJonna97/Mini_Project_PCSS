import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('mortenilna.png')

cv.imshow("Original", img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Binary_inv sets the minimum and max value. The OTSU is an algorithm to choose the best threshold.
ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_TRIANGLE)

cv.imshow("Black and white", thresh)

# Remove noise
kernel = np.ones((3, 3),np.uint8)
# Erosion of the image.
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

# sure background area
# Dilation of the image
sure_bg = cv.dilate(opening, kernel, iterations=3)

# Finding sure foreground are
# distanceTransform makes the background black, but apparently white in my case.
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
ret, sure_fg = cv.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
# Calculates the per-element difference between two arrays or array and a scalar.
unknown = cv.subtract(sure_bg, sure_fg)

cv.imshow("distant transform", dist_transform)

# Marker labelling
# Computes the connected components labeled image.
ret, markers = cv.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1

# Now mark the region of unknown with zero
markers[unknown == 255] = 0

# marker-based watershed algorithm, where the foreground objects are marked as peaks,
# and the rest are background or non-objects.
markers = cv.watershed(img, markers)
img[markers == -1] = [255, 0, 0]

cv.imshow("Result", img)
cv.waitKey(0)
cv.destroyAllWindows()
