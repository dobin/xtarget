import cv2
import numpy as np
import logging
import copy

from model import OpencvRect, RecordedHit

logger = logging.getLogger(__name__)


class Projector():
    def __init__(self, height=1920, width=1080):
        self.frame = np.zeros((width,height,3), np.uint8)
        self.recordedHit = None

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
        self.offsetRadius = None

        self.colorTarget = (200, 0, 0)
        self.colorHit = (200, 0, 0)


    def draw(self):
        frame = self.frame.copy()

        cv2.circle(self.frame, 
            (self.projectorTargetCenterX, self.projectorTargetCenterY), 
            self.projectorTargetRadius, 
            self.colorTarget, 10)

        if self.camTargetCenterX != None:
            cv2.circle(self.frame, 
                ( int(self.camTargetCenterX * self.offsetX), int(self.camTargetCenterY * self.offsetY)), 
                int(self.camTargetRadius * self.offsetRadius), 
                (100, 100, 0), 5)

        if self.recordedHit != None:
            cv2.circle(self.frame, (self.recordedHit.x, self.recordedHit.y), self.recordedHit.radius, self.colorHit, 4)
        cv2.imshow('Projector', frame)


    def setTargetCenter(self, x, y, targetRadius):
        logger.info("Set target center: {} /  {} radius {}".format(x, y, targetRadius))
        self.camTargetCenterX = int(x)
        self.camTargetCenterY = int(y)
        self.camTargetRadius = int(targetRadius)

        self.offsetX = self.projectorTargetCenterX / self.camTargetCenterX
        self.offsetY = self.projectorTargetCenterY / self.camTargetCenterY
        self.offsetRadius = self.projectorTargetRadius / self.camTargetRadius

        logger.info("  Offset: {} / {}  radius {}".format(self.offsetX, self.offsetY, self.offsetRadius))


    def handleShot(self, recordedHit):
        # check if shot hit something
        #  to make target disappear
        recordedHit = copy.copy(recordedHit)
        logging.info("A: {} / {} - {}".format(recordedHit.x, recordedHit.y, recordedHit.radius))

        recordedHit.x = int(recordedHit.x * self.offsetX)
        recordedHit.y = int(recordedHit.y * self.offsetY)
        recordedHit.radius = int(recordedHit.radius * self.offsetRadius)
        
        logging.info("B: {} / {} - {}".format(recordedHit.x, recordedHit.y, recordedHit.radius))

        self.recordedHit = recordedHit
