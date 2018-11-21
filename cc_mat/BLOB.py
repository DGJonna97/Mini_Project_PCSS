import numpy as np
import cv2
import math

class BLOB:

    #Python class constructor
    def __init__(self):
        self.pixels = [] #Stored as an array of "points" aka. x,y coordinates
        self.binary = None
        self.area = None
        self.centerOfMass = None
        self.rect = None
        self.compactness = None
        self.circularity = None
        self.perimeter = None

    def setPixels(self, pixels):
        self.pixels = pixels

    def addPixel(self, pixel):
        self.pixels.append(pixel)

    #Function for getting the pixels as a binary image (Instead of one list of pixels)
    def getBinaryImg(self):
        #Only calculate the binary image if it has not yet been calculated
        if(self.binary is None):
            x, y, w, h = self.getRect()

            #Initiating the empty image from w and h (+3 is to create some padding)
            self.binary = np.zeros((h+3, w+3), np.uint8)

            #putting in white pixels in respective locations
            for p in self.pixels:
                self.binary[p[1]-y+1][p[0]-x+1] = 255 #I do -2 to add the padding mentioned earlier

        return self.binary


    def getArea(self):
        if(self.area is None):
            self.area = len(self.pixels)

        return self.area


    def getCenterOfMass(self):
        if(self.centerOfMass is None):

            #This will split the array in two, one with xpositions and on with y positions
            xArr, yArr = np.hsplit(np.array(self.pixels), 2)

            #Doing the summation based on the formula in the IP book
            xCom = (1/len(self.pixels)) * np.sum(xArr)
            yCom = (1/len(self.pixels)) * np.sum(yArr)

            #Sets center of mass x and y position
            self.centerOfMass = [int(xCom), int(yCom)]

        return self.centerOfMass[0], self.centerOfMass[1]

    def getRect(self): #Essentially the bounding box
        if(self.rect is None):

            x = 8000
            y = 8000
            w = 0
            h = 0

            #Loops though the pixel array to find the largest / smallest x and y value
            for p in self.pixels:
                if p[0] < x:
                    x = p[0]
                if p[0] > w:
                    w = p[0]
                if p[1] < y:
                    y = p[1]
                if p[1] > h:
                    h = p[1]

            self.rect = [x, y, w-x, h-y]

        #returns array in the following order (x, y, w, h)
        return self.rect[0], self.rect[1], self.rect[2], self.rect[3]

    def getCompactness(self):
        if(self.compactness is None):
            _, _, w, h = self.getRect()
            area = self.getArea()

            #Based on the fomula in the IP book
            self.compactness = area / (w*h)

        return self.compactness

    def getPerimeter(self):
        if(self.perimeter is None):
            binImg = self.getBinaryImg()

            image,contours,_ = cv2.findContours(binImg, 1, 2)

            perimeter = 0;

            for x in contours:
                    perimeter += cv2.arcLength(contours[0], True)

            self.perimeter = perimeter

        return self.perimeter

    def getCircularity(self):
        if(self.circularity is None):
            #Based off the function in the IP book
            self.circularity = self.getPerimeter() / (2 * math.sqrt(math.pi * self.getArea()))

        return self.circularity

#Function for getting blobs from the component Image
#Returns an array of the BLOB object described above
def getBlobs(components):

    componentArray = np.empty((np.max(components)), dtype=object)

    for x in range(len(componentArray)):
        componentArray[x] = BLOB()

    for y in range(len(components)):
        for x in range(len(components[y])):
            if(components[y][x] != 0):
                componentArray[components[y][x]-1].addPixel([x, y])

    return componentArray
