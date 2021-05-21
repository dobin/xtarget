import cv2 as cv
import imutils
import yaml
import logging

from gfxutils import *
from model import *


# from ball_tracking.py
def findContours(mask, minRadius):
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
            logging.info("Found dot with radius " + str(radius) + "at  X:" + str(x) + "  Y:" + str(y))

            recordedHit = RecordedHit()
            recordedHit.x = int(x)
            recordedHit.y = int(y)
            recordedHit.center = center
            recordedHit.radius = int(radius)
            res.append(recordedHit)
        else:
            logging.info("Too small: " + str(radius))
            pass

    return res
    

class Lazer(object):
    def __init__(self, videoStream, thresh=14, saveFrames=False, saveHits=False, mode=Mode.main):
        self.capture = None
        self.frame = None
        self.mask = None
        self.previousMask = None
        self.lastFoundFrameNr = 0
        self.mode = mode
        self.debug = True

        self.videoStream = videoStream
        self.thresh = thresh

        # debug options
        self.saveFrames = saveFrames
        self.saveHits = saveHits

        # decoding options
        self.minRadius = 1.0
        self.doDenoise = True
        self.doSharpen = True

        # data for target
        self.centerX = 0
        self.centerY = 0
        self.hitRadius = 0

        # detection options
        self.graceTime = 10  # How many frames between detections
        
        self.resetDynamic()


    def resetDynamic(self):
        self.glareMeter = 0
        self.glareMeterAvg = 0
        self.hits = []
        self.lastFoundFrameNr = 0


    def changeMode(self, mode):
        self.mode = mode


    def getDistanceToCenter(self, x, y):
        return calculateDistance(self.centerX, self.centerY, x, y)

    
    def setCenter(self, x, y, hitRadius):
        logging.info("Set center: " + str(x) + " / " + str(y))
        self.centerX = int(x)
        self.centerY = int(y)
        self.hitRadius = int(hitRadius)


    def nextFrame(self):
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
        if not staticImage:
            # wait a bit between detections
            if (self.videoStream.frameNr - self.lastFoundFrameNr) < self.graceTime:
                return []

            # check if there is any change at all
            # if no change, do not attempt to find contours. 
            # this can save processing power
            if frameIdentical(self.mask, self.previousMask):
                return []

        recordedHits = findContours(self.mask, self.minRadius)
        if len(recordedHits) > 0:
            self.lastFoundFrameNr = self.videoStream.frameNr
            logging.debug("Found hit at frame #" + str(self.videoStream.frameNr) + " with radius " + str(recordedHits[0].radius))
        else:
            return []

        # augment frame with hit indicators
        if staticImage or self.mode == Mode.main:
            for recordedHit in recordedHits:
                # draw
                cv.circle(self.frame, (int(recordedHit.x), int(recordedHit.y)), int(recordedHit.radius), (0, 100, 50), 2)
                cv.circle(self.frame, recordedHit.center, 5, (0, 250, 50), -1)

                # check if we have a target (to measure distance to)
                if self.hitRadius > 0:
                    p = int(self.getDistanceToCenter(recordedHit.x, recordedHit.y))
                    r = self.hitRadius
                    d = int(p/r * 100)
                    recordedHit.distance = d

                self.hits.append(recordedHit)

                if self.saveHits:
                    self.saveCurrentFrame(recordedHit)

        return recordedHits


    def checkGlare(self):
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

        if self.centerX != 0:
            cv.circle(self.frame, (self.centerX, self.centerY), self.hitRadius, (0,200,0), 2)

        cv.imshow('Video', self.frame)
        cv.imshow('Mask', self.mask)


    def saveCurrentFrame(self, recordedHit=None):
        filenameBase = self.videoStream.getFilenameBase()
        filenameBase += '_'  + str(self.videoStream.frameNr) + '_'

        logging.info("Saving current frame:")
        if recordedHit != None:
            filenameBase += 'hit.'
            fname = filenameBase + "info.yaml"
            logging.info("  Save yaml to : " + fname)
            with open(fname, 'w') as outfile:
                yaml.dump(recordedHit.toDict(), outfile)

        fname = filenameBase + 'frame.jpg'
        logging.info("  Save Frame to: " + fname)
        cv.imwrite(fname, self.frame)

        fname = filenameBase + 'mask.jpg' 
        logging.info("  Save Mask to : " + fname)
        cv.imwrite(fname, self.mask)


    def release(self):
        self.videoStream.release()
