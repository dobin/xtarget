import cv2
import yaml
import logging

from gfxutils import *
from model import *
from detectorthread import DetectorThread
from projector import Projector

logger = logging.getLogger(__name__)


class Lazer(object):
    """Manages detection via Detector on VideoStream"""

    def __init__(self, videoStream, thresh=14, saveFrames=False, saveHits=False, mode=Mode.main):
        self.saveFrames = saveFrames
        self.saveHits = saveHits
        self.detectorThread = DetectorThread(videoStream)
        self.projector = Projector()

        self.debug = True
        self.frameNr = None
        self.glareEnabled = False  # TODO

        # static hit options
        self.hitGraceTime = 30  # How many frames between detections (~1s)
        self.hitMinRadius = 1.0  # found out by experimentation

        self.resetDynamic()

        # read/written by Lazer
        # only read by DetectorThread
        self.threadData = {
            'mode': mode,
            'thresh': thresh,
            'targetThresh': 60,  # going up, looked good with testing
        }
        self.detectorThread.startThread(self.threadData)


    def resetDynamic(self):
        """resets dynamic parameter used to track temporary things (ui cleanup)"""
        self.glareMeter = 0
        self.glareMeterAvg = 0
 
        self.hits = []
        self.hitLastFoundFrameNr = 0  # Track when last hit was found

        # data for identified target
        self.targetCenterX = None
        self.targetCenterY = None
        self.targetRadius = None
        self.noAutoTarget = False

        # data for aruco
        self.arucoCorners = None
        self.arucoIds = None


    def addThresh(self, thresh):
        self.threadData['thresh'] += thresh


    def getThresh(self):
        return self.threadData['thresh']


    def changeMode(self, mode):
        self.threadData['mode'] = mode
        print("new mode: " + str(self.threadData['mode']))
        if mode == Mode.main:
            self.projector.setTargetCenter(self.targetCenterX, self.targetCenterY, self.targetRadius)
            self.projector.setCamAruco(self.arucoCorners, self.arucoIds)
        elif mode == Mode.intro:
            self.resetDynamic()


    def getMode(self):
        return self.threadData['mode']


    def getDistanceToCenter(self, x, y):
        return calculateDistance(self.targetCenterX, self.targetCenterY, x, y)

    
    def setTargetCenter(self, x, y, targetRadius):
        """Sets and enabled the target"""
        logger.info("Manually set target center: " + str(x) + " / " + str(y))
        self.targetCenterX = int(x)
        self.targetCenterY = int(y)
        self.targetRadius = int(targetRadius)
        self.noAutoTarget = True


    def nextFrame(self):
        """Retrieves next frame from video/cam via VideoStream, process it and store into self.frame and self.mask"""
        isTrue, self.frame, self.frameNr, mode, data = self.detectorThread.Q.get()
        if not isTrue:  # end of file or stream
            return False, None

        # reset stats if file rewinds
        if self.frameNr == 0:
            self.resetDynamic()

        if mode == Mode.intro:
            self.handleGlare(data['glare'])
            self.handleTarget(data['targetContours'], data['targetReliefs'])
            self.handleAruco(data['arucoCorners'], data['arucoIds'], data['arucoRejected'])
        elif mode == Mode.main:
            data['recordedHits'] = self.handleHits(data['recordedHits'])

        # if we wanna record everything
        if self.saveFrames:
            self.saveCurrentFrame()

        return True, data


    def handleAruco(self, corners, ids, rejected):
        if self.arucoCorners != None:
            return

        for corner in corners:
            a = (int(corner[0][0][0]), int(corner[0][0][1]))
            b = (int(corner[0][2][0]), int(corner[0][2][1]))
            cv2.rectangle(self.frame, 
                a,
                b,
                (0,255,255), 2)

        if len(corners) != 4:
            return

        self.arucoCorners = corners
        self.arucoIds = ids
        logger.info("Found 4 aruco {} {}".format(len(corners), len(ids)))
        self.projector.setCamAruco(self.arucoCorners, self.arucoIds)


    def drawAruco(self):
        self.projector.initFrame()
        if self.threadData['mode'] == Mode.intro:
            if self.arucoCorners != None:
                # draw aruco area
                a = (int(self.arucoCorners[3][0][0][0]), int(self.arucoCorners[3][0][0][1]))
                b = (int(self.arucoCorners[0][0][2][0]), int(self.arucoCorners[0][0][2][1]))
                cv2.rectangle(self.frame, 
                    a,
                    b,
                    (0,255,255), 2)
            else:
                self.projector.drawAruco()
        elif self.threadData['mode'] == Mode.main:
            self.projector.drawTargetCircle()
            self.projector.drawHits()


    def handleTarget(self, contours, reliefs, save=False):
        return
        if self.threadData['targetThresh'] > 150:
            # give up here
            return
        if self.targetCenterX != None:
            return
        if self.noAutoTarget:
            return

        for relief in reliefs:
            cv2.circle(self.frame, (relief.centerX, relief.centerY), 10, (100, 255, 100), -1)
            for c in contours:
                cv2.drawContours(self.frame, [c], -1, (0, 255, 0), 2)

        if len(reliefs) == 0:
            self.threadData['targetThresh'] += 1
        else:
            print("Auto Target at {}/{} with thresh {}".format(reliefs[0].centerX, reliefs[0].centerY, self.threadData['targetThresh']))
            self.targetCenterX = reliefs[0].centerX
            self.targetCenterY = reliefs[0].centerY
            self.targetRadius = int(reliefs[0].w / 2)


    def drawTarget(self):
        cv2.circle(self.frame, (self.targetCenterX, self.targetCenterY), self.targetRadius, (0,200,0), 2)


    def handleGlare(self, glare):
        if not self.glareEnabled:
            return

        for rect in glare:
            cv2.rectangle(self.frame, (rect.x, rect.y), (rect.x + rect.w, rect.y + rect.h), (0, 0, 255), 2)

        if len(glare) > 0:
            if self.glareMeter < 60:  # 30 is typical fps, so 1s
                self.glareMeter += 4
        else:
            if self.glareMeter > 0:
                self.glareMeter -= 1
        self.glareMeterAvg = int(self.glareMeterAvg + self.glareMeter) >> 1


    def handleHits(self, recordedHits, staticImage=False):
        if not staticImage and self.hitLastFoundFrameNr != 0:
            # wait a bit between detections
            if (self.frameNr - self.hitLastFoundFrameNr) < self.hitGraceTime:
                return []
        
        if len(recordedHits) > 0:
            self.hitLastFoundFrameNr = self.frameNr
            logger.info("Found hit at frame #" + str(self.frameNr) + " with radius " + str(recordedHits[0].radius))
            self.projector.handleShot(recordedHits[0])

        # draw
        for recordedHit in recordedHits:
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

        return recordedHits


    def displayFrame(self):
        """Displays the current frame in the window, with UI data written on it"""

        # Stuff we found out
        if self.targetCenterX != None:
            self.drawTarget()
        self.drawAruco()

        # UI
        o = 300
        color = (255, 255, 255)
        s= "Tresh: " + str(self.getThresh())
        cv2.putText(self.frame, s, (o*0,30), cv2.FONT_HERSHEY_TRIPLEX, 1.0, color, 2)

        if self.glareMeterAvg > 0:
            cv2.putText(self.frame, "Glare: " + str(self.glareMeterAvg), (0,140), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 0, 255), 2)

        s = "Mode: " + str(self.threadData['mode'].name)
        cv2.putText(self.frame, s, (o*0,90), cv2.FONT_HERSHEY_TRIPLEX, 1.0, color, 2)
        if self.detectorThread.videoStream.fps.get() < 28:
            s = "FPS: " + str(self.detectorThread.videoStream.fps.get())
            cv2.putText(self.frame, s, (o*1,90), cv2.FONT_HERSHEY_TRIPLEX, 1.0, color, 2)

        if self.debug:
            s = 'Frame: '+ str(self.frameNr)
            cv2.putText(self.frame, s, (o*1,30), cv2.FONT_HERSHEY_TRIPLEX, 1.0, color, 2)
            #s = "Denoise: " + str(self.detector.doDenoise)
            #cv2.putText(self.frame, s, ((o*0),60), cv2.FONT_HERSHEY_TRIPLEX, 1.0, color, 2)
            #s = "Sharpen: " + str(self.detector.doSharpen)
            #cv2.putText(self.frame, s, (o*1,60), cv2.FONT_HERSHEY_TRIPLEX, 1.0, color, 2)

            s = "Target: {}/{} {}".format(self.targetCenterX, self.targetCenterY, self.targetRadius)
            cv2.putText(self.frame, s, (o*1,120), cv2.FONT_HERSHEY_TRIPLEX, 1.0, color, 2)

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
        if self.threadData['mode'] == Mode.intro:
            s = "Press SPACE to start"
            cv2.putText(self.frame, s, ((self.detectorThread.videoStream.width >> 1) - 60,self.detectorThread.videoStream.height - 30), cv2.FONT_HERSHEY_TRIPLEX, 1.0, color, 2)        
        elif self.threadData['mode'] == Mode.main:
            s = "Press SPACE to stop"
            cv2.putText(self.frame, s, ((self.detectorThread.videoStream.width >> 1) - 60,self.detectorThread.videoStream.height - 30), cv2.FONT_HERSHEY_TRIPLEX, 1.0, color, 2)        

        # draw
        cv2.imshow('Video', self.frame)
        if self.debug:
            #cv2.imshow('Mask', self.detector.mask)
            pass
        self.projector.show()


    def saveCurrentFrame(self, recordedHit=None):
        """Save current frame as file"""
        filenameBase = self.videoStream.getFilenameBase()
        filenameBase += '_'  + str(self.frameNr) + '_'

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
        self.detectorThread.shutdownThread()
