import cv2
import numpy as np
import logging

from model import OpencvRect

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


    def draw(self):
        frame = self.frame.copy()

        cv2.circle(self.frame, (self.projectorTargetCenterX, self.projectorTargetCenterY), self.projectorTargetRadius, (255,255,255), 20)
        if self.recordedHit != None:
            cv2.circle(self.frame, 
                (int(self.recordedHit.x * self.offsetX), int(self.recordedHit.y * self.offsetY)), 
                int(self.recordedHit.radius * self.offsetRadius), 
                (255,255,255), 4)
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


    def handleShot(self, recordedHit):
        # check if shot hit something
        #  to make target disappear
        self.recordedHit = recordedHit
