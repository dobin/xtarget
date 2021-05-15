import math
import numpy as np


def frameIdentical(image1, image2):
    return image1.shape == image2.shape and not(np.bitwise_xor(image1,image2).any())

def calculateDistance(x1,y1,x2,y2):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist
