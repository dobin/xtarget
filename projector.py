import cv2
import numpy as np
import logging

from numpy.lib.arraysetops import isin

from model import OpencvRect, RecordedHit
from gfxutils import imageCopyInto

logger = logging.getLogger(__name__)


class Projector():
    def __init__(self, height=1080, width=1920):
        self.recordedHit = None
        self.width = width
        self.height = height

        # what we project
        self.projectorTargetCenterX = 700
        self.projectorTargetCenterY = 600
        self.projectorTargetRadius = 100

        # input from webcam
        self.camTargetCenterX = None
        self.camTargetCenterY = None
        self.camTargetRadius = None
        self.offsetX = None
        self.offsetY = None

        # colors
        self.colorTarget = (200, 0, 0)
        self.colorHit = (0, 100, 0)
        self.colorAruco = (255, 255, 255)

        # aruco target area we project
        self.arucoX = self.projectorTargetCenterX - 250
        self.arucoY = self.projectorTargetCenterY - 250
        self.arucoWidth = 500
        self.arucoHeight = 500
        # aruco library options
        self.arucoSymbolSize = 100
        self.arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
        # aruca symbols
        l = self.arucoSymbolSize
        self.arucoA = cv2.aruco.drawMarker(self.arucoDict, 42, l)
        self.arucoB = cv2.aruco.drawMarker(self.arucoDict, 1001, l)
        self.arucoC = cv2.aruco.drawMarker(self.arucoDict, 241, l)
        self.arucoD = cv2.aruco.drawMarker(self.arucoDict, 1007, l)

        # aruco cam->projector transform
        self.H = None
        self.srcMat = None
        self.dstMat = None

        self.frameSrc = np.zeros((height,width,3), np.uint8)
        self.picAruco = self._getPicAruco()
        self.picTargetBase = self._getPicTarget()


    def showTarget(self):
        cv2.imshow('Projector', self.picTargetBase)


    def showAruco(self):
        cv2.imshow('Projector', self.picAruco)


    def handleShot(self, recordedHit):
        if self.H is None:
            return

        x, y = self.translate(recordedHit.x, recordedHit.y)

        print("Shot at   : {}/{}".format(recordedHit.x, recordedHit.y))
        print("Shot trans1: {}/{}".format(x, y))
        print("Shot trans2: {}/{}".format(x+self.arucoX, y+self.arucoY))

        self.recordedHit = RecordedHit()
        self.recordedHit.x = x+self.arucoX
        self.recordedHit.y = y+self.arucoY
        self.recordedHit.radius = recordedHit.radius

        cv2.circle(self.picTargetBase, 
            (self.recordedHit.x, self.recordedHit.y), 20, 
            self.colorHit, 4)



    def setTargetCenter(self, x, y, targetRadius):
        if x is None or y is None:
            return
        logger.info("Set target center: " + str(x) + " / " + str(y))
        self.camTargetCenterX = int(x)
        self.camTargetCenterY = int(y)
        self.camTargetRadius = int(targetRadius)

        self.offsetX = (self.projectorTargetCenterX / self.camTargetCenterX)
        self.offsetY = (self.projectorTargetCenterY / self.camTargetCenterY)
        self.offsetRadius = (self.projectorTargetRadius / self.camTargetRadius)


    def setCamAruco(self, arucoCorners, arucoIds):
        """Set the detected Cam/Video Arucos, and calculate the transformation matrix to us"""
        if arucoCorners == None:
            return
        ids = arucoIds.flatten()
        refPts = []
        
        for i in (42, 1001, 241, 1007):
            if len(np.where(ids == i)) == 0:
                print("Not found: {}".format(i))
                return

        # loop over the IDs of the ArUco markers in top-left, top-right,
        # bottom-right, and bottom-left order
        for i in (42, 1001, 241, 1007):
            # grab the index of the corner with the current ID and append the
            # corner (x, y)-coordinates to our list of reference points
            j = np.squeeze(np.where(ids == i))

            print("-> i: {}  j: {}".format(i, j))
            if isinstance(j, list):
                print("-> Something went wrong")
                return
            
            corner = np.squeeze(arucoCorners[j])
            refPts.append(corner)

        # unpack our ArUco reference points and use the reference points to
        # define the *destination* transform matrix, making sure the points
        # are specified in top-left, top-right, bottom-right, and bottom-left
        # order
        (refPtTL, refPtTR, refPtBR, refPtBL) = refPts
        dstMat = [refPtTL[0], refPtTR[1], refPtBR[2], refPtBL[3]]
        dstMat = np.array(dstMat)

        # grab the spatial dimensions of the source image and define the
        # transform matrix for the *source* image in top-left, top-right,
        # bottom-right, and bottom-left order
        srcMat = np.array([[0, 0], [self.arucoWidth, 0], [self.arucoWidth, self.arucoHeight], [0, self.arucoHeight]])
        # compute the homography matrix
        (H, _) = cv2.findHomography(srcMat, dstMat)

        print("H: " + str(H))
        self.H = H
        self.srcMat = srcMat
        self.dstMat = dstMat


    def translate(self, x, y):
        H_inv = np.linalg.inv(self.H)
        p = np.array((x,y,1)).reshape((3,1))
        temp_p = H_inv.dot(p)
        sum = np.sum(temp_p ,1)
        px = int(round(sum[0]/sum[2]))
        py = int(round(sum[1]/sum[2]))
        return px, py


    def _getPicTarget(self):
        """Draw the actual target circle we project"""
        frame = self.frameSrc.copy()

        # the actual target circle
        cv2.circle(frame, 
            (self.projectorTargetCenterX, self.projectorTargetCenterY), 
            self.projectorTargetRadius, 
            self.colorTarget, 10)

        return frame


    def _getPicAruco(self):
        """Draw the four Aruco squares we project"""
        frame = self.frameSrc.copy()
        c = (200, 200, 200)

        # aruco
        # ?? When running the detection on a single marker, the results are best when 
        # ?? the size of the white margin is at least as big as the black border of the marker.
        lineWidth = 30
        lineHalfWidth = (lineWidth >> 1) + 0
        #cv2.rectangle(frame, 
        #    (self.arucoX, self.arucoY), 
        #    (self.arucoX+self.arucoWidth, self.arucoY+self.arucoHeight),
        #    (0, 255, 0), 1)

        # this is stupid
        # A
        imageCopyInto(frame, self.arucoA, 
            self.arucoX, 
            self.arucoY
        )
        cv2.rectangle(frame,
            (self.arucoX-lineHalfWidth, self.arucoY-lineHalfWidth), 
            (self.arucoX+self.arucoSymbolSize+lineHalfWidth, self.arucoY+self.arucoSymbolSize+lineHalfWidth),
            c, lineWidth
        )
        # B
        imageCopyInto(frame, self.arucoB, 
            self.arucoX + self.arucoWidth - self.arucoSymbolSize, 
            self.arucoY
        )
        cv2.rectangle(frame,
            (self.arucoX-lineHalfWidth + self.arucoWidth - self.arucoSymbolSize, self.arucoY-lineHalfWidth), 
            (self.arucoX + self.arucoWidth - self.arucoSymbolSize+self.arucoSymbolSize+lineHalfWidth, self.arucoY+self.arucoSymbolSize+lineHalfWidth),
            c, lineWidth
        )
        # C
        imageCopyInto(frame, self.arucoC, 
            self.arucoX + self.arucoWidth - self.arucoSymbolSize, 
            self.arucoY + self.arucoHeight - self.arucoSymbolSize
        )
        cv2.rectangle(frame,
            (self.arucoX-lineHalfWidth + self.arucoWidth - self.arucoSymbolSize, self.arucoY-lineHalfWidth+ self.arucoHeight - self.arucoSymbolSize), 
            (self.arucoX+self.arucoSymbolSize+lineHalfWidth + self.arucoWidth - self.arucoSymbolSize, self.arucoY+self.arucoSymbolSize+lineHalfWidth+ self.arucoHeight - self.arucoSymbolSize),
            c, lineWidth
        )
        # D
        imageCopyInto(frame, self.arucoD, 
            self.arucoX, 
            self.arucoY + self.arucoHeight - self.arucoSymbolSize
        )
        cv2.rectangle(frame,
            (self.arucoX-lineHalfWidth, self.arucoY-lineHalfWidth+ self.arucoHeight - self.arucoSymbolSize), 
            (self.arucoX+self.arucoSymbolSize+lineHalfWidth, self.arucoY+self.arucoSymbolSize+lineHalfWidth+ self.arucoHeight - self.arucoSymbolSize),
            c, lineWidth
        )

        return frame