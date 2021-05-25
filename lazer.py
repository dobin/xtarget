import cv2
import yaml
import logging

from gfxutils import *
from model import *
from detector import Detector
from projector import Projector

logger = logging.getLogger(__name__)


class Lazer(object):
    """Manages detection via Detector on VideoStream"""

    def __init__(self, videoStream, thresh=14, saveFrames=False, saveHits=False, mode=Mode.main):
        self.videoStream = videoStream
        self.saveFrames = saveFrames
        self.saveHits = saveHits
        self.mode = mode
        self.detector = Detector(thresh=thresh)
        self.projector = Projector()
        self.debug = True

        # static hit options
        self.hitGraceTime = 30  # How many frames between detections (1s)
        self.hitMinRadius = 1.0  # found out by experimentation

        self.resetDynamic()


    def resetDynamic(self):
        """resets dynamic parameter used to track temporary things (ui cleanup)"""
        self.glareMeter = 0
        self.glareMeterAvg = 0
 
        self.hits = []
        self.hitLastFoundFrameNr = 0  # Track when last hit was found

       # data for identified target
        self.targetThresh = 60  # hopyfully sane initial value, going up
        self.targetCenterX = None
        self.targetCenterY = None
        self.targetRadius = None


    def changeMode(self, mode):
        self.mode = mode
        self.detector.mode = mode
        if self.mode == Mode.main:
            # take over target, if any
            self.handleTarget(save=True)


    def getDistanceToCenter(self, x, y):
        return calculateDistance(self.targetCenterX, self.targetCenterY, x, y)

    
    def setTargetCenter(self, x, y, targetRadius):
        """Sets and enabled the target"""
        logger.debug("Set target center: " + str(x) + " / " + str(y))
        self.targetCenterX = int(x)
        self.targetCenterY = int(y)
        self.targetRadius = int(targetRadius)


    def nextFrame(self):
        """Retrieves next frame from video/cam via VideoStream, process it and store into self.frame and self.mask"""
        isTrue, self.frame = self.videoStream.getFrame()
        if not isTrue:  # end of file or stream
            return False
        
        # reset stats if file rewinds
        if self.videoStream.frameNr == 0:
            self.resetDynamic()

        self.detector.initFrame(frame=self.frame)
        if self.mode == Mode.intro:
            self.handleGlare()
            self.handleTarget()
        elif self.mode == Mode.main:
            recordedHits = self.getHits()
            self.drawHits(recordedHits)

        # if we wanna record everything
        if self.saveFrames:
            self.saveCurrentFrame()

        return True


    def handleTarget(self, save=False):
        if self.targetThresh > 150:
            # give up here
            return

        contours, reliefs = self.detector.findTargets(self.targetThresh)
        for relief in reliefs:
            cv2.circle(self.frame, (relief.centerX, relief.centerY), 10, (100, 255, 100), -1)
            for c in contours:
                cv2.drawContours(self.frame, [c], -1, (0, 255, 0), 2)

        if len(reliefs) == 0:
            self.targetThresh += 1
        elif save:
            print("Target at {}/{} with thresh {}".format(reliefs[0].centerX, reliefs[0].centerY, self.targetThresh))
            self.targetCenterX = reliefs[0].centerX
            self.targetCenterY = reliefs[0].centerY
            self.targetRadius = int(reliefs[0].w / 2)
            self.projector.setTargetCenter(self.targetCenterX, self.targetCenterY, self.targetRadius)


    def handleGlare(self):
        glare = self.detector.findGlare()
        for rect in glare:
            cv2.rectangle(self.frame, (rect.x, rect.y), (rect.x + rect.w, rect.y + rect.h), (0, 0, 255), 2)

        if len(glare) > 0:
            if self.glareMeter < 60:  # 30 is typical fps, so 1s
                self.glareMeter += 4
        else:
            if self.glareMeter > 0:
                self.glareMeter -= 1
        self.glareMeterAvg = int(self.glareMeterAvg + self.glareMeter) >> 1


    def getHits(self, staticImage=False):
        if not staticImage:
            # wait a bit between detections
            if (self.videoStream.frameNr - self.hitLastFoundFrameNr) < self.hitGraceTime:
                return []

        recordedHits = self.detector.findHits(self.hitMinRadius)
        if len(recordedHits) > 0:
            self.hitLastFoundFrameNr = self.videoStream.frameNr
            logger.debug("Found hit at frame #" + str(self.videoStream.frameNr) + " with radius " + str(recordedHits[0].radius))
            self.projector.handleShot(recordedHits[0])

        return recordedHits


    def drawHits(self, recordedHits):
        for recordedHit in recordedHits:
            # draw
            cv2.circle(self.frame, (int(recordedHit.x), int(recordedHit.y)), int(recordedHit.radius), (0, 100, 50), 2)
            cv2.circle(self.frame, recordedHit.center, 5, (0, 250, 50), -1)

            # check if we have a target (to measure distance to)
            if self.targetRadius != None:
                p = int(self.getDistanceToCenter(recordedHit.x, recordedHit.y))
                r = self.targetRadius
                d = int(p/r * 100)
                recordedHit.distance = d

            self.hits.append(recordedHit)

            if self.saveHits:
                self.saveCurrentFrame(recordedHit)


    def displayFrame(self):
        """Displays the current frame in the window, with UI data written on it"""
        o = 300

        color = (255, 255, 255)
        s= "Tresh: " + str(self.detector.thresh)
        cv2.putText(self.frame, s, (o*0,30), cv2.FONT_HERSHEY_TRIPLEX, 1.0, color, 2)

        if self.glareMeterAvg > 0:
            cv2.putText(self.frame, "Glare: " + str(self.glareMeterAvg), (0,140), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 0, 255), 2)

        s = "Mode: " + str(self.mode.name)
        cv2.putText(self.frame, s, (o*0,90), cv2.FONT_HERSHEY_TRIPLEX, 1.0, color, 2)
        if self.videoStream.fps.get() < 28:
            s = "FPS: " + str(self.videoStream.fps.get())
            cv2.putText(self.frame, s, (o*1,90), cv2.FONT_HERSHEY_TRIPLEX, 1.0, color, 2)

        if self.debug:
            s = 'Frame: '+ str(self.videoStream.frameNr)
            cv2.putText(self.frame, s, (o*1,30), cv2.FONT_HERSHEY_TRIPLEX, 1.0, color, 2)
            s = "Denoise: " + str(self.detector.doDenoise)
            cv2.putText(self.frame, s, ((o*0),60), cv2.FONT_HERSHEY_TRIPLEX, 1.0, color, 2)
            s= "Sharpen: " + str(self.detector.doSharpen)
            cv2.putText(self.frame, s, (o*1,60), cv2.FONT_HERSHEY_TRIPLEX, 1.0, color, 2)

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

            cv2.putText(self.frame, s, (0,0+140+(30*idx)), cv2.FONT_HERSHEY_TRIPLEX, 1.0, color, 2)
            cv2.circle(self.frame, (hit.x, hit.y), hit.radius, color, 2)
            cv2.circle(self.frame, (hit.x, hit.y), 10, color, -1)

        color = (0, 0, 255)
        if self.mode == Mode.intro:
            s = "Press SPACE to start"
            cv2.putText(self.frame, s, ((self.videoStream.width >> 1) - 60,self.videoStream.height - 30), cv2.FONT_HERSHEY_TRIPLEX, 1.0, color, 2)        
        elif self.mode == Mode.main:
            s = "Press SPACE to stop"
            cv2.putText(self.frame, s, ((self.videoStream.width >> 1) - 60,self.videoStream.height - 30), cv2.FONT_HERSHEY_TRIPLEX, 1.0, color, 2)        

        if self.targetCenterX != None:
            cv2.circle(self.frame, (self.targetCenterX, self.targetCenterY), self.targetRadius, (0,200,0), 2)

        cv2.imshow('Video', self.frame)
        if self.debug:
            cv2.imshow('Mask', self.detector.mask)
        self.projector.draw()


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
        cv2.imwrite(fname, self.frame)

        fname = filenameBase + 'mask.jpg' 
        logger.info("  Save Mask to : " + fname)
        cv2.imwrite(fname, self.mask)


    def release(self):
        self.videoStream.release()

