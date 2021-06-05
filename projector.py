import cv2
import numpy as np
import logging

from model import OpencvRect

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

        # Pentone
        self.pentoneX = self.projectorTargetCenterX - 100
        self.pentoneY = self.projectorTargetCenterY - 100
        self.pentoneWidth = 179
        self.pentoneHeight = 227

        self.H = None
        self.srcMat = None
        self.dstMat = None
        

    def draw(self):
        frame = self.frame.copy()

        # pantone
        cv2.rectangle(frame, 
            (self.pentoneX, self.pentoneY), 
            (self.pentoneX+self.pentoneWidth, self.pentoneY+self.pentoneHeight),
            (0, 255, 0), 3)

        cv2.circle(frame, 
            (self.projectorTargetCenterX, self.projectorTargetCenterY), 
            self.projectorTargetRadius, 
            self.colorTarget, 10)

        #if self.recordedHit != None:
        #    cv2.circle(frame, 
        #        (int(self.recordedHit.x * self.offsetX), int(self.recordedHit.y * self.offsetY)), 
        #        int(self.recordedHit.radius * self.offsetRadius), 
        #        self.colorHit, 4)

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


    def setPanton(self, pantonCorners, pantonIds):
        ids = pantonIds.flatten()
        refPts = []
        # loop over the IDs of the ArUco markers in top-left, top-right,
        # bottom-right, and bottom-left order
        for i in (923, 1001, 241, 1007):
            # grab the index of the corner with the current ID and append the
            # corner (x, y)-coordinates to our list of reference points
            j = np.squeeze(np.where(ids == i))
            corner = np.squeeze(pantonCorners[j])
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
        srcMat = np.array([[0, 0], [self.pentoneWidth, 0], [self.pentoneWidth, self.pentoneHeight], [0, self.pentoneHeight]])
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
        print("Shot trans2: {}/{}".format(x+self.pentoneX, y+self.pentoneY))
        cv2.circle(self.frame, 
            (x+self.pentoneX, y+self.pentoneY), recordedHit.radius, 
            self.colorHit, 4)

        #cv2.circle(self.frame, 
        #    (50, 50), 4, 
        #    self.colorHit, 4)

        #cv2.circle(self.frame, 
        #    (int(self.recordedHit.x * self.offsetX), int(self.recordedHit.y * self.offsetY)), 
        #    int(self.recordedHit.radius * self.offsetRadius), 
        #    self.colorHit, 4)



    def translate(self, x, y):
        H_inv = np.linalg.inv(self.H)
        #H_inv = self.H

        p = np.array((x,y,1)).reshape((3,1))
        temp_p = H_inv.dot(p)
        sum = np.sum(temp_p ,1)
        px = int(round(sum[0]/sum[2]))
        py = int(round(sum[1]/sum[2]))

        return px, py