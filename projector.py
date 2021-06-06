import cv2
import numpy as np
import logging

from model import OpencvRect
from gfxutils import imageCopyInto

logger = logging.getLogger(__name__)


class Projector():
    def __init__(self, height=1080, width=1920):
        self.frame = np.zeros((height,width,3), np.uint8)
        self.recordedHit = None
        self.width = width
        self.height = height

        # what we project
        self.projectorTargetCenterX = 700
        self.projectorTargetCenterY = 400
        self.projectorTargetRadius = 100

        # input from webcam
        self.camTargetCenterX = None
        self.camTargetCenterY = None
        self.camTargetRadius = None
        self.offsetX = None
        self.offsetY = None

        # colors
        self.colorTarget = (200, 0, 0)
        self.colorHit = (0, 200, 0)
        self.colorAruco = (255, 255, 255)

        # Aruco
        self.arucoX = self.projectorTargetCenterX - 250
        self.arucoY = self.projectorTargetCenterY - 250
        self.arucoWidth = 500
        self.arucoHeight = 500
        self.arucoSymbolSize = 100
        self.arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)

        # Aruco Transform
        self.H = None
        self.srcMat = None
        self.dstMat = None

        self.initAruco()
        

    def initAruco(self):
        l = self.arucoSymbolSize
        self.arucoA = cv2.aruco.drawMarker(self.arucoDict, 923, l)
        self.arucoB = cv2.aruco.drawMarker(self.arucoDict, 1001, l)
        self.arucoC = cv2.aruco.drawMarker(self.arucoDict, 241, l)
        self.arucoD = cv2.aruco.drawMarker(self.arucoDict, 1007, l)

        
    def draw(self):
        frame = self.frame.copy()

        # aruco
        lineWidth = 10
        lineHalfWidth = lineWidth >> 1
        cv2.rectangle(frame, 
            (self.arucoX, self.arucoY), 
            (self.arucoX+self.arucoWidth, self.arucoY+self.arucoHeight),
            (0, 255, 0), 1)

        # this is stupid
        # A
        imageCopyInto(frame, self.arucoA, 
            self.arucoX, 
            self.arucoY
        )
        cv2.rectangle(frame,
            (self.arucoX-lineHalfWidth, self.arucoY-lineHalfWidth), 
            (self.arucoX+self.arucoSymbolSize+lineHalfWidth, self.arucoY+self.arucoSymbolSize+lineHalfWidth),
            (255, 255, 255), 10
        )
        # B
        imageCopyInto(frame, self.arucoB, 
            self.arucoX + self.arucoWidth - self.arucoSymbolSize, 
            self.arucoY
        )
        cv2.rectangle(frame,
            (self.arucoX-lineHalfWidth + self.arucoWidth - self.arucoSymbolSize, self.arucoY-lineHalfWidth), 
            (self.arucoX + self.arucoWidth - self.arucoSymbolSize+self.arucoSymbolSize+lineHalfWidth, self.arucoY+self.arucoSymbolSize+lineHalfWidth),
            (255, 255, 255), 10
        )
        # C
        imageCopyInto(frame, self.arucoC, 
            self.arucoX + self.arucoWidth - self.arucoSymbolSize, 
            self.arucoY + self.arucoHeight - self.arucoSymbolSize
        )
        cv2.rectangle(frame,
            (self.arucoX-lineHalfWidth + self.arucoWidth - self.arucoSymbolSize, self.arucoY-lineHalfWidth+ self.arucoHeight - self.arucoSymbolSize), 
            (self.arucoX+self.arucoSymbolSize+lineHalfWidth + self.arucoWidth - self.arucoSymbolSize, self.arucoY+self.arucoSymbolSize+lineHalfWidth+ self.arucoHeight - self.arucoSymbolSize),
            (255, 255, 255), 10
        )
        # D
        imageCopyInto(frame, self.arucoD, 
            self.arucoX, 
            self.arucoY + self.arucoHeight - self.arucoSymbolSize
        )
        cv2.rectangle(frame,
            (self.arucoX-lineHalfWidth, self.arucoY-lineHalfWidth+ self.arucoHeight - self.arucoSymbolSize), 
            (self.arucoX+self.arucoSymbolSize+lineHalfWidth, self.arucoY+self.arucoSymbolSize+lineHalfWidth+ self.arucoHeight - self.arucoSymbolSize),
            (255, 255, 255), 10
        )

        # the actual target circle
        cv2.circle(frame, 
            (self.projectorTargetCenterX, self.projectorTargetCenterY), 
            self.projectorTargetRadius, 
            self.colorTarget, 10)

        cv2.imshow('Projector', frame)


    def setTargetCenter(self, x, y, targetRadius):
        logger.info("Set target center: " + str(x) + " / " + str(y))
        self.camTargetCenterX = int(x)
        self.camTargetCenterY = int(y)
        self.camTargetRadius = int(targetRadius)

        self.offsetX = (self.projectorTargetCenterX / self.camTargetCenterX)
        self.offsetY = (self.projectorTargetCenterY / self.camTargetCenterY)
        self.offsetRadius = (self.projectorTargetRadius / self.camTargetRadius)

        logging.info("OffsetX: " + str(self.offsetX))


    def setAruco(self, arucoCorners, arucoIds):
        ids = arucoIds.flatten()
        refPts = []
        # loop over the IDs of the ArUco markers in top-left, top-right,
        # bottom-right, and bottom-left order
        for i in (923, 1001, 241, 1007):
            # grab the index of the corner with the current ID and append the
            # corner (x, y)-coordinates to our list of reference points
            j = np.squeeze(np.where(ids == i))
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


    def handleShot(self, recordedHit):
        # check if shot hit something
        #  to make target disappear
        self.recordedHit = recordedHit

        print("Shot at   : {}/{}".format(recordedHit.x, recordedHit.y))
        x, y = self.translate(recordedHit.x, self.recordedHit.y)
        print("Shot trans1: {}/{}".format(x, y))
        print("Shot trans2: {}/{}".format(x+self.arucoX, y+self.arucoY))
        cv2.circle(self.frame, 
            (x+self.arucoX, y+self.arucoY), recordedHit.radius, 
            self.colorHit, 4)


    def translate(self, x, y):
        H_inv = np.linalg.inv(self.H)
        #H_inv = self.H

        p = np.array((x,y,1)).reshape((3,1))
        temp_p = H_inv.dot(p)
        sum = np.sum(temp_p ,1)
        px = int(round(sum[0]/sum[2]))
        py = int(round(sum[1]/sum[2]))

        return px, py