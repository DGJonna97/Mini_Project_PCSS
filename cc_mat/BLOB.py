import numpy as np

class BLOB:

    def __init__(self):
        self.pixels = []

    def setPixels(self, pixels):
        self.pixels = pixels

    def addPixel(self, pixel):
        self.pixels.append(pixel)


    def getArea(self):
        return len(self.pixels)

    def getRect():
        return np.array([0,0,0,0])

def getBlobs(components):

    componentArray = np.empty((np.max(components)), dtype=object)

    for x in range(len(componentArray)):
        componentArray[x] = BLOB()

    for y in range(len(components)):
        for x in range(len(components[y])):
            componentArray[components[y][x]-1].addPixel([x, y])

    return componentArray
