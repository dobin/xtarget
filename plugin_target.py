import logging
import cv2

logger = logging.getLogger(__name__)


class PluginTarget(object):
    def __init__(self):
        # all public
        self.targetCenterX = None
        self.targetCenterY = None
        self.targetRadius = None

        self.noAutoTarget = False
        self.targetThreshStart = 100
        self.targetThreshEnd = 300
        self.targetThresh = self.targetThreshStart
        self.reliefCounter = 0


    def init(self):
        self.targetCenterX = None
        self.targetCenterY = None
        self.targetRadius = None
        self.noAutoTarget = False
        self.relief = None
        self.reliefCounter = 0


    def handle(self, frame, contours, reliefs, save=False):
        # draw it always
        #for relief in reliefs:
        for c in contours:
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)

        if self.relief is not None:
            cv2.circle(frame, (self.relief.centerX, self.relief.centerY), 10, (100, 255, 100), -1)

        # do nothing when already set
        if self.targetCenterX is not None:
            return

        if not self.noAutoTarget:
            # reset if necessary
            if self.targetThresh > self.targetThreshEnd:
                self.targetThresh = self.targetThreshStart

            # adjust threshhold
            if len(reliefs) == 0:
                self.targetThresh += 1

        if len(reliefs) > 0:
            if self.reliefCounter > 20:  # all 20 frames with reliefs identified
                self.reliefCounter = 0
                self.relief = reliefs[0]  # just take a random one, its all circles
            else:
                self.reliefCounter += 1


    def useCurrentTarget(self):
        self.targetCenterX = self.relief.centerX
        self.targetCenterY = self.relief.centerY
        self.targetRadius = int(self.relief.w / 2)


    def draw(self, frame):
        if self.targetCenterX is not None:
            cv2.circle(frame, (self.targetCenterX, self.targetCenterY), self.targetRadius, (0, 200, 0), 2)


    def setTargetCenter(self, x, y, targetRadius):
        """Sets and enabled the target"""
        logger.info("Manually set target center: " + str(x) + " / " + str(y))
        self.targetCenterX = int(x)
        self.targetCenterY = int(y)
        self.targetRadius = int(targetRadius)
        self.noAutoTarget = True
