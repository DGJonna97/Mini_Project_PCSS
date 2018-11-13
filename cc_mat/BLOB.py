import numpy as np
import cv2
import math

class BLOB:

    def __init__(self):
        self.pixels = []
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

    def getBinaryImg(self):
        if(self.binary is None):
            x, y, w, h = self.getRect()

            self.binary = np.zeros((abs(h)+3, abs(w)+3), np.uint8)

            for p in self.pixels:
                self.binary[p[1]-y-2][p[0]-x-2] = 255

        return self.binary


    def getArea(self):
        if(self.area is None):
            self.area = len(self.pixels)

        return self.area

    def getCenterOfMass(self):
        if(self.centerOfMass is None):
            xArr, yArr = np.hsplit(np.array(self.pixels), 2)

            xCom = (1/len(self.pixels)) * np.sum(xArr)
            yCom = (1/len(self.pixels)) * np.sum(yArr)

            self.centerOfMass = [int(xCom), int(yCom)]

        return self.centerOfMass[0], self.centerOfMass[1]

    def getRect(self):
        if(self.rect is None):

            x = 0
            y = 0
            w = 8000
            h = 8000

            for p in self.pixels:
                if p[0] > x:
                    x = p[0]
                if p[0] < w:
                    w = p[0]
                if p[1] > y:
                    y = p[1]
                if p[1] < h:
                    h = p[1]

            self.rect = [x, y, w-x, h-y]

        return self.rect[0], self.rect[1], self.rect[2], self.rect[3]

    def getCompactness(self):
        if(self.compactness is None):
            _, _, w, h = self.getRect()
            area = self.getArea()

            self.compactness = area / (w*h)

        return self.compactness

    def getPerimeter(self):
        if(self.perimeter is None):
            binImg = self.getBinaryImg()

            kernel = np.ones((3, 3), np.uint8)

            binImg_small = cv2.erode(binImg, kernel, iterations=1)

            edgeImg = np.subtract(binImg, binImg_small)

            cv2.imshow("perimeter", edgeImg)

            self.perimeter = np.sum(edgeImg) / 255

        return self.perimeter

    def getCircularity(self):
        if(self.circularity is None):
            self.circularity = self.getPerimeter() / (2 * math.sqrt(math.pi * self.getArea()))

        return self.circularity




def getBlobs(components):

    componentArray = np.empty((np.max(components)), dtype=object)

    for x in range(len(componentArray)):
        componentArray[x] = BLOB()

    for y in range(len(components)):
        for x in range(len(components[y])):
            componentArray[components[y][x]-1].addPixel([x, y])

    return componentArray[0:-1]
