import cv2
import imutils

from gfxutils import *
from model import *

class Detector():
    def __init__(self, videoStream):
        self.videoStream = videoStream

        # decoding options
        self.doDenoise = True
        self.doSharpen = True

        self.arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.arucoParams = cv2.aruco.DetectorParameters_create()
        self.init()


    def init(self):
        self.frame = None
        self.grey = None
        self.mask = None
        self.previousMask = None
        self.mask2 = None
        

    def initFrame(self, frame, thresh):
        self.frame = frame
        self.previousMask = self.mask

        # Mask: Make to grey
        self.grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ##self.mask3 = self.mask.copy()

        # Mask: remove small artefacts (helpful for removing some glare, and improving detection)
        if self.doSharpen:
            m = cv2.medianBlur(self.grey,5)
            # self.mask = cv2.blur(self.mask,(5,5))
            m = cv2.erode(m, (7,7), iterations=3)

        # save copy of mask for now
        #self.mask2 = self.mask.copy()
        self.mask2 = m
        # Mask: threshold, throw away all bytes below thresh (bytes)
        _, self.mask = cv2.threshold(m, 255-thresh, 255, cv2.THRESH_BINARY)

        ##cv2.imshow("m3", self.mask2)
        
        # check if there is any change at all
        # if no change, do not attempt to find contours. 
        # this can save processing power
        #if frameIdentical(self.mask, self.previousMask):
        #    return []


    def findAruco(self):
        (corners, ids, rejected) = cv2.aruco.detectMarkers(self.mask2, self.arucoDict, parameters=self.arucoParams)
        return (corners, ids, rejected)


    def findGlare(self):
        """Check self.frame for glare, and highlight it"""

        # glare should be detected on mask - it needs to be clean
        cnts = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # loop over the contours
        glare = []
        for c in cnts:
            # compute the bounding box of the contour and then draw the
            # bounding box on both input images to represent where the two
            # images differ
            (x, y, w, h) = cv2.boundingRect(c)
            opencvRect = OpencvRect(x, y, w, h)
            glare.append(opencvRect)
            
        return glare


    def findTargets(self, targetThresh):
        thresh = cv2.threshold(self.mask2, targetThresh, 255, cv2.THRESH_BINARY_INV)[1]
        cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # use RETR_TREE here to get all
        cnts = imutils.grab_contours(cnts)
        reliefs = []
        contours = []
        for c in cnts:
            relief = findCircles(c)
            if relief == None:
                continue
            reliefs.append(relief)
            contours.append(c)

        return contours, reliefs


    def findHits(self, minRadius):
        res = []

        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            if M["m00"] == 0:
                ##print("DIVISION BY ZERO")
                return res
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # only proceed if the radius meets a minimum size
            if radius > minRadius:  # orig: 10, for most: 5
                radius = int(radius)
                x = int(x)
                y = int(y)
                logger.debug("Found dot with radius " + str(radius) + " at  X:" + str(x) + "  Y:" + str(y))

                recordedHit = RecordedHit()
                recordedHit.x = x
                recordedHit.y = y
                recordedHit.center = center
                recordedHit.radius = radius
                res.append(recordedHit)
            else:
                logger.info("Too small: " + str(radius))
                pass

        return res


def findTriangles(c):
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)

    # if the shape is a triangle, it will have 3 vertices
    if len(approx) != 3:
        return None

    # compute the bounding box of the contour and use the
    # bounding box to compute the aspect ratio
    (x, y, w, h) = cv2.boundingRect(approx)

    # probably too small
    if w < 100 or h < 100:
        return None

    # not similar height/width
    ar = w / float(h)
    if ar < 0.9 or ar > 1.3:
        return None

    relief = Relief()
    relief.x = x
    relief.y = y
    relief.w = w
    relief.h = h

    # compute the center of the contour, then detect the name of the
    # shape using only the contour
    M = cv2.moments(c)
    if M["m00"] != 0:
        cX = int((M["m10"] / M["m00"]))
        cY = int((M["m01"] / M["m00"]))
        relief.centerX = cX
        relief.centerY = cY
    else:
        print("Division by zero")

    return relief


def findCircles(c):
    # initialize the shape name and approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)

    # lots of vertices pls
    if len(approx) < 7:
        return None

    # compute the bounding box of the contour and use the
    # bounding box to compute the aspect ratio
    (x, y, w, h) = cv2.boundingRect(approx)

    # probably too small
    if w < 50 or h < 50:
        return None

    relief = Relief()
    relief.x = x
    relief.y = y
    relief.w = w
    relief.h = h

    # compute the center of the contour
    M = cv2.moments(c)
    if M["m00"] != 0:
        cX = int((M["m10"] / M["m00"]))
        cY = int((M["m01"] / M["m00"]))
        relief.centerX = cX
        relief.centerY = cY
    else:
        print("Division by zero")

    return relief