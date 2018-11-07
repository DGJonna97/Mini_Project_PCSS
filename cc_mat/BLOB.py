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

    def getRect(self):

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

        return x, y, w-x, h-y

def getBlobs(components):

    componentArray = np.empty((np.max(components)), dtype=object)

    for x in range(len(componentArray)):
        componentArray[x] = BLOB()

    for y in range(len(components)):
        for x in range(len(components[y])):
            componentArray[components[y][x]-1].addPixel([x, y])

    return componentArray