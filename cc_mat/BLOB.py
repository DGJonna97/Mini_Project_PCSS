import numpy as np
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
        if(self.binary == None):
            x, y, w, h = self.getRect()

            self.binary = np.zeros((abs(w), abs(h)), np.bool_)

            for p in self.pixels:
                self.binary[p[0]-x][p[1]-y] = 1

        return self.binary


    def getArea(self):
        if(self.area == None):
            self.area = len(self.pixels)

        return self.area

    def getCenterOfMass(self):
        if(self.centerOfMass == None):
            xArr, yArr = np.hsplit(np.array(self.pixels), 2)

            xCom = (1/len(self.pixels)) * np.sum(xArr)
            yCom = (1/len(self.pixels)) * np.sum(yArr)

            self.centerOfMass = [int(xCom), int(yCom)]

        return self.centerOfMass[0], self.centerOfMass[1]

    def getRect(self):
        if(self.rect == None):

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
        if(self.compactness == None):
            _, _, w, h = self.getRect()
            area = self.getArea()

            self.compactness = area / (w*h)

        return self.compactness

    def getPerimeter(self):
        if(self.perimeter == None):
            binImg = self.getBinaryImg()

            counter = 0

            for y in range(len(binImg)):
                for x in range(len(binImg[y])):
                    if(binImg[y][x] == 1):

                        connectedSides = 0

                        if(y-1 > 0):
                            if(binImg[y-1][x] == 1):
                                connectedSides += 1
                        if(x-1 > 0):
                            if(binImg[y][x-1] == 1):
                                connectedSides += 1
                        if(y+1 < len(binImg)):
                            if(binImg[y+1][x] == 1):
                                connectedSides += 1
                        if(x+1 < len(binImg[y])):
                            if(binImg[y][x+1] == 1):
                                connectedSides += 1



                        if(connectedSides < 4):
                            counter += 1

            self.perimeter = counter

        return self.perimeter

    def getCircularity(self):
        if(self.circularity == None):
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
