import cv2 as cv
import imutils
import yaml
import logging

from gfxutils import *
from model import *

logger = logging.getLogger(__name__)


class Lazer(object):
    """Detect and visualize hits in a frame with a mask every time nextFrame() is called"""

    def __init__(self, videoStream, thresh=14, saveFrames=False, saveHits=False, mode=Mode.main):
        self.videoStream = videoStream
        self.thresh = thresh
        self.saveFrames = saveFrames
        self.saveHits = saveHits
        self.mode = mode

        self.frame = None  # from nextFrame(): the frame from the cam/video
        self.mask = None  # from nextFrame(): mask generated based on frame
        self.previousMask = None  # mask of previous iteration of the nextFrame() loop
        self.debug = True

        # decoding options
        self.minRadius = 1.0
        self.doDenoise = True
        self.doSharpen = True

        # data for target
        self.targetCenterX = 0
        self.targetCenterY = 0
        self.targetHitRadius = 0

        # detection options
        self.graceTime = 10  # How many frames between detections
        self.lastFoundHitFrameNr = 0  # Track when last hit was found

        self.resetDynamic()


    def resetDynamic(self):
        """resets dynamic parameter used to track temporary things (ui cleanup)"""
        self.glareMeter = 0
        self.glareMeterAvg = 0
        self.hits = []
        self.lastFoundHitFrameNr = 0


    def changeMode(self, mode):
        self.mode = mode


    def getDistanceToCenter(self, x, y):
        return calculateDistance(self.targetCenterX, self.targetCenterY, x, y)

    
    def setTargetCenter(self, x, y, targetHitRadius):
        """Sets and enabled the target"""
        logger.debug("Set target center: " + str(x) + " / " + str(y))
        self.targetCenterX = int(x)
        self.targetCenterY = int(y)
        self.targetHitRadius = int(targetHitRadius)


    def nextFrame(self):
        """Retrieves next frame from video/cam via VideoStream, process it and store into self.frame and self.mask"""
        self.previousMask = self.mask

        isTrue, self.frame = self.videoStream.getFrame()
        if not isTrue:  # end of file or stream
            return False

        # reset stats if file rewinds
        if self.videoStream.frameNr == 0:
            self.resetDynamic()

        # Mask: Make to grey
        self.mask = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)

        # Mask: remove small artefacts (helpful for removing some glare, and improving detection)
        if self.doSharpen:
            self.mask = cv.medianBlur(self.mask,5)
            # self.mask = cv.blur(self.mask,(5,5))
            self.mask = cv.erode(self.mask, (7,7), iterations=3)
        
        # Mask: threshold, throw away all bytes below thresh (bytes)
        _, self.mask = cv.threshold(self.mask, 255-self.thresh, 255, cv.THRESH_BINARY)

        # Mask: Check if there is glare and handle it (glaremeter, drawing rectangles)
        if self.mode == Mode.intro:
            self.checkGlare()

        # if we wanna record everything
        if self.saveFrames:
            self.saveCurrentFrame()

        return True


    def detectAndDrawHits(self, staticImage=False):
        """Use self.mask to detect hits in the picture and highlight them"""
        if not staticImage:
            # wait a bit between detections
            if (self.videoStream.frameNr - self.lastFoundHitFrameNr) < self.graceTime:
                return []

            # check if there is any change at all
            # if no change, do not attempt to find contours. 
            # this can save processing power
            if frameIdentical(self.mask, self.previousMask):
                return []

        recordedHits = findHits(self.mask, self.minRadius)
        if len(recordedHits) > 0:
            self.lastFoundHitFrameNr = self.videoStream.frameNr
            logger.debug("Found hit at frame #" + str(self.videoStream.frameNr) + " with radius " + str(recordedHits[0].radius))
        else:
            return []

        # augment frame with hit indicators
        if staticImage or self.mode == Mode.main:
            for recordedHit in recordedHits:
                # draw
                cv.circle(self.frame, (int(recordedHit.x), int(recordedHit.y)), int(recordedHit.radius), (0, 100, 50), 2)
                cv.circle(self.frame, recordedHit.center, 5, (0, 250, 50), -1)

                # check if we have a target (to measure distance to)
                if self.targetHitRadius > 0:
                    p = int(self.getDistanceToCenter(recordedHit.x, recordedHit.y))
                    r = self.targetHitRadius
                    d = int(p/r * 100)
                    recordedHit.distance = d

                self.hits.append(recordedHit)

                if self.saveHits:
                    self.saveCurrentFrame(recordedHit)

        return recordedHits


    def checkGlare(self):
        """Check self.frame for glare, and highlight it"""
        # threshold the difference image, followed by finding contours to
        # obtain the regions of the two input images that differ
        #thresh = cv.threshold(diff, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
        #cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        cnts = cv.findContours(self.mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # loop over the contours
        for c in cnts:
            # compute the bounding box of the contour and then draw the
            # bounding box on both input images to represent where the two
            # images differ
            (x, y, w, h) = cv.boundingRect(c)
            cv.rectangle(self.frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            #cv.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

        if len(cnts) > 0:
            if self.glareMeter < 60:  # 30 is typical fps, so 1s
                self.glareMeter += 4
        else:
            if self.glareMeter > 0:
                self.glareMeter -= 1
        self.glareMeterAvg = int(self.glareMeterAvg + self.glareMeter) >> 1


    def displayFrame(self):
        """Displays the current frame in the window, with UI data written on it"""
        o = 300

        color = (255, 255, 255)
        s= "Tresh: " + str(self.thresh)
        cv.putText(self.frame, s, (o*0,30), cv.FONT_HERSHEY_TRIPLEX, 1.0, color, 2)

        if self.glareMeterAvg > 0:
            cv.putText(self.frame, "Glare: " + str(self.glareMeterAvg), (0,140), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0, 0, 255), 2)

        s = "Mode: " + str(self.mode.name)
        cv.putText(self.frame, s, (o*0,90), cv.FONT_HERSHEY_TRIPLEX, 1.0, color, 2)
        if self.videoStream.fps.get() < 28:
            s = "FPS: " + str(self.videoStream.fps.get())
            cv.putText(self.frame, s, (o*1,90), cv.FONT_HERSHEY_TRIPLEX, 1.0, color, 2)

        if self.debug:
            s = 'Frame: '+ str(self.videoStream.frameNr)
            cv.putText(self.frame, s, (o*1,30), cv.FONT_HERSHEY_TRIPLEX, 1.0, color, 2)
            s = "Denoise: " + str(self.doDenoise)
            cv.putText(self.frame, s, ((o*0),60), cv.FONT_HERSHEY_TRIPLEX, 1.0, color, 2)
            s= "Sharpen: " + str(self.doSharpen)
            cv.putText(self.frame, s, (o*1,60), cv.FONT_HERSHEY_TRIPLEX, 1.0, color, 2)

        for idx, hit in enumerate(self.hits): 
            if hit.distance > 0:
                s = str(idx) + " distance: " + str(hit.distance) + " (r:" + str(hit.radius) + ")"
            else:
                s = str(idx) + " (r:" + str(hit.radius) + ")"
            if idx == 0:
                color = (0, 200, 0)
            elif idx == 1:
                color = (0, 100, 240)
            elif idx == 2:
                color = (150, 0, 200)
            else:
                color = (0, 170, 200)

            cv.putText(self.frame, s, (0,0+140+(30*idx)), cv.FONT_HERSHEY_TRIPLEX, 1.0, color, 2)
            cv.circle(self.frame, (hit.x, hit.y), hit.radius, color, 2)
            cv.circle(self.frame, (hit.x, hit.y), 10, color, -1)

        color = (0, 0, 255)
        if self.mode == Mode.intro:
            s = "Press SPACE to start"
            cv.putText(self.frame, s, ((self.videoStream.width >> 1) - 60,self.videoStream.height - 30), cv.FONT_HERSHEY_TRIPLEX, 1.0, color, 2)        
        elif self.mode == Mode.main:
            s = "Press SPACE to stop"
            cv.putText(self.frame, s, ((self.videoStream.width >> 1) - 60,self.videoStream.height - 30), cv.FONT_HERSHEY_TRIPLEX, 1.0, color, 2)        

        if self.targetCenterX != 0:
            cv.circle(self.frame, (self.targetCenterX, self.targetCenterY), self.targetHitRadius, (0,200,0), 2)

        cv.imshow('Video', self.frame)
        cv.imshow('Mask', self.mask)


    def saveCurrentFrame(self, recordedHit=None):
        """Save current frame as file"""
        filenameBase = self.videoStream.getFilenameBase()
        filenameBase += '_'  + str(self.videoStream.frameNr) + '_'

        logger.info("Saving current frame:")
        if recordedHit != None:
            filenameBase += 'hit.'
            fname = filenameBase + "info.yaml"
            logger.info("  Save yaml to : " + fname)
            with open(fname, 'w') as outfile:
                yaml.dump(recordedHit.toDict(), outfile)

        fname = filenameBase + 'frame.jpg'
        logger.info("  Save Frame to: " + fname)
        cv.imwrite(fname, self.frame)

        fname = filenameBase + 'mask.jpg' 
        logger.info("  Save Mask to : " + fname)
        cv.imwrite(fname, self.mask)


    def release(self):
        self.videoStream.release()


# from ball_tracking.py
def findHits(mask, minRadius):
    res = []

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv.contourArea)
        ((x, y), radius) = cv.minEnclosingCircle(c)
        M = cv.moments(c)
        if M["m00"] == 0:
            ##print("DIVISION BY ZERO")
            return res
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
        if radius > minRadius:  # orig: 10, for most: 5
            radius = int(radius)
            x = int(x)
            y = int(y)
            logger.info("Found dot with radius " + str(radius) + "at  X:" + str(x) + "  Y:" + str(y))

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
