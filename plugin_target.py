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
        self.targetThresh = 60


    def init(self):
        self.targetCenterX = None
        self.targetCenterY = None
        self.targetRadius = None
        self.noAutoTarget = False
        self.targetThresh = 60


    def handle(self, frame, contours, reliefs, save=False):
        if self.targetThresh > 150:
            # give up here
            return
        if self.targetCenterX is not None:
            return
        if self.noAutoTarget:
            return

        for relief in reliefs:
            cv2.circle(frame, (relief.centerX, relief.centerY), 10, (100, 255, 100), -1)
            for c in contours:
                cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)

        if len(reliefs) == 0:
            self.targetThresh += 1
        else:
            print("Auto Target at {}/{} with thresh {}".format(reliefs[0].centerX, reliefs[0].centerY, self.targetThresh))
            self.targetCenterX = reliefs[0].centerX
            self.targetCenterY = reliefs[0].centerY
            self.targetRadius = int(reliefs[0].w / 2)


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
