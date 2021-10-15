import logging
import cv2

logger = logging.getLogger(__name__)


class PluginHits(object):
    def __init__(self):
        self.hitLastFoundFrameNr = 0  # Track when last hit was found
        self.hitGraceTime = 30  # How many frames between detections (~1s)

    def init(self):
        self.hitLastFoundFrameNr = 0  # Track when last hit was found
        self.hitGraceTime = 30  # How many frames between detections (~1s)


    def handle(self, frame, frameNr, recordedHits, staticImage=False):
        """Check if new hits have been detected in the frame"""
        if not staticImage and self.hitLastFoundFrameNr != 0:
            # wait a bit between detections
            if (frameNr - self.hitLastFoundFrameNr) < self.hitGraceTime:
                return None

        if len(recordedHits) == 0:
            return None
        if len(recordedHits) != 1:
            logger.warn("More than one hit!!!")

        hit = recordedHits[0]

        # for some things, we dont care how many we detected, just that we detected at least one
        logger.info("Found hit at frame #" + str(frameNr) + " with radius " + str(recordedHits[0].radius))
        self.hitLastFoundFrameNr = frameNr
        
        self.draw(frame, hit)

        return hit


    def draw(self, frame, recordedHit):
        cv2.circle(frame, (int(recordedHit.x), int(recordedHit.y)), int(recordedHit.radius), (0, 100, 50), 2)
        cv2.circle(frame, recordedHit.center, 5, (0, 250, 50), -1)

