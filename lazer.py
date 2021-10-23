import cv2
import yaml
import logging
import numpy as np

from gfxutils import calculateDistance
from model import Mode
from detectorthread import DetectorThread
from projector import Projector
from gamemode import GameMode
from plugin_hits import PluginHits
from plugin_glare import PluginGlare
from plugin_target import PluginTarget
from plugin_aruco import PluginAruco

logger = logging.getLogger(__name__)



class Lazer(object):
    """Manages detection via Detector on VideoStream"""

    def __init__(
        self, videoStream, thresh=14,
        withProjector=False, saveFrames=False, saveHits=False, mode=Mode.main, enableTarget=False, debug=True
    ):
        self.saveFrames = saveFrames
        self.saveHits = saveHits
        self.detectorThread = DetectorThread(videoStream)
        self.projector = None
        self.withProjector = withProjector
        self.projector = Projector()
        self.gameMode = GameMode()

        self.debug = debug
        self.glareEnabled = True
        self.targetEnabled = enableTarget

        self.pluginHits = PluginHits()
        self.pluginGlare = PluginGlare()
        self.pluginTarget = PluginTarget()
        self.pluginAruco = PluginAruco()

        # per frame
        self.frame = None
        self.frameNr = 0
        self.mode = None

        self.resetDynamic()

        # read/written by Lazer
        # only read by DetectorThread
        self.threadData = {
            'mode': mode,  # model.Mode
            'thresh': thresh,  # threshold for mask in bit
            'targetThresh': 60,  # going up, looked good with testing
            'crop': None,  # when input image from the webcam should be cropped
        }
        self.detectorThread.startThread(self.threadData)


    def resetDynamic(self):
        """resets dynamic parameter used to track temporary things (ui cleanup)"""
        self.hits = []

        self.pluginGlare.init()
        self.pluginHits.init()
        self.pluginTarget.init()
        self.pluginAruco.init()


    def nextFrame(self):
        """Retrieves next frame from video/cam via VideoStream, process it and draw into self.frame"""
        isTrue, self.frame, self.frameNr, self.mode, data = self.detectorThread.getFrameData()
        if not isTrue:  # end of file or stream
            return False, None

        # reset stats if file rewinds
        if self.frameNr == 0:
            self.resetDynamic()

        self.gameMode.nextFrame(self.frameNr)

        if self.mode == Mode.intro:
            if self.glareEnabled:
                self.pluginGlare.handle(self.frame, data['glare'])
            if self.targetEnabled:
                self.pluginTarget.handle(self.frame, data['targetContours'], data['targetReliefs'])
                self.threadData["targetThresh"] = self.pluginTarget.targetThresh
            if self.withProjector:
                self.pluginAruco.handle(self.frame, data['arucoCorners'], data['arucoIds'], data['arucoRejected'])
                if len(self.pluginAruco.arucoCornersAll) == 4 and self.projector.H is None:
                    corners = np.array(list(self.pluginAruco.arucoCornersAll.values()))
                    ids = np.array(list(self.pluginAruco.arucoIdsAll.values()))
                    self.projector.setCamAruco(corners, ids)
        elif self.mode == Mode.main:
            self.handleMain(self.frame, self.frameNr, data['recordedHits'])

        # if we wanna record everything
        if self.saveFrames:
            self.saveCurrentFrame()

        return True, data


    def handleMain(self, frame, frameNr, recordedHits):
        hit = self.pluginHits.handle(frame, frameNr, recordedHits)
        if hit is not None:
            # check if we have a target (to measure distance to)
            if self.pluginTarget.targetRadius is not None:
                p = int(calculateDistance(self.pluginTarget.targetCenterX, self.pluginTarget.targetCenterY, hit.x, hit.y))
                r = self.pluginTarget.targetRadius
                d = int((p / r) * 100)
                hit.distance = d

            hit.time = self.gameMode.reset()
            self.hits.append(hit)

            if self.saveHits:
                self.saveCurrentFrame(hit)


    def displayFrame(self):
        """Displays the current frame in the window, with UI data written on it"""
        if self.targetEnabled:
            self.pluginTarget.draw(self.frame)
        if self.withProjector and self.mode == Mode.intro:
            self.pluginAruco.draw(self.frame)
        self.drawUi()
        self.drawHits()
        self.drawGameMode()

        cv2.imshow('Video', self.frame)
        if self.debug:
            #cv2.imshow('Mask', self.detector.mask)
            pass
        if self.withProjector:
            if self.threadData['mode'] == Mode.intro:
                self.projector.showAruco()
            elif self.threadData['mode'] == Mode.main:
                self.projector.showTarget()


    def changeMode(self, mode):
        self.threadData['mode'] = mode
        print("new mode: " + str(self.threadData['mode']))
        if mode == Mode.main:
            self.pluginTarget.useCurrentTarget()
            self.gameMode.start()
            if self.withProjector:
                self.projector.setTargetCenter(
                    self.pluginTarget.targetCenterX, self.pluginTarget.targetCenterY, self.pluginTarget.targetRadius)
                self.projector.setCamAruco(self.pluginAruco.arucoCorners, self.pluginAruco.arucoIds)
        elif mode == Mode.intro:
            self.gameMode.stop()
            self.resetDynamic()


    def drawUi(self):
        # UI
        o = 300
        color = (255, 255, 255)
        s = "Tresh: " + str(self.getThresh()) + " / " + str(self.pluginTarget.targetThresh - 100)
        cv2.putText(self.frame, s, (o * 0, 30), cv2.FONT_HERSHEY_TRIPLEX, 1.0, color, 2)

        if self.pluginGlare.glareMeterAvg > 0:
            cv2.putText(
                self.frame,
                "Glare: " + str(self.pluginGlare.glareMeterAvg),
                (0, 140),
                cv2.FONT_HERSHEY_TRIPLEX,
                1.0,
                (0, 0, 255), 2)

        s = "Mode: " + str(self.threadData['mode'].name)
        cv2.putText(self.frame, s, (o * 0, 90), cv2.FONT_HERSHEY_TRIPLEX, 1.0, color, 2)
        if self.detectorThread.videoStream.fps.get() < 28:
            s = "FPS: " + str(self.detectorThread.videoStream.fps.get())
            cv2.putText(self.frame, s, (o * 1, 90), cv2.FONT_HERSHEY_TRIPLEX, 1.0, color, 2)

        if self.debug:
            s = 'Frame: ' + str(self.frameNr)
            cv2.putText(self.frame, s, (o * 1, 30), cv2.FONT_HERSHEY_TRIPLEX, 1.0, color, 2)
            #s = "Denoise: " + str(self.detector.doDenoise)
            #cv2.putText(self.frame, s, ((o*0),60), cv2.FONT_HERSHEY_TRIPLEX, 1.0, color, 2)
            #s = "Sharpen: " + str(self.detector.doSharpen)
            #cv2.putText(self.frame, s, (o*1,60), cv2.FONT_HERSHEY_TRIPLEX, 1.0, color, 2)

            s = "Target: {}/{} {}".format(self.pluginTarget.targetCenterX, self.pluginTarget.targetCenterY, self.pluginTarget.targetRadius)
            cv2.putText(self.frame, s, (o * 1, 120), cv2.FONT_HERSHEY_TRIPLEX, 1.0, color, 2)

        # hints
        color = (0, 0, 255)
        if self.threadData['mode'] == Mode.intro:
            s = "Press SPACE to start"
            cv2.putText(
                self.frame,
                s,
                ((self.detectorThread.videoStream.width >> 1) - 60, self.detectorThread.videoStream.height - 30),
                cv2.FONT_HERSHEY_TRIPLEX,
                1.0,
                color,
                2)
        elif self.threadData['mode'] == Mode.main:
            s = "Press SPACE to stop"
            cv2.putText(
                self.frame,
                s,
                ((self.detectorThread.videoStream.width >> 1) - 60, self.detectorThread.videoStream.height - 30),
                cv2.FONT_HERSHEY_TRIPLEX,
                1.0,
                color,
                2)


    def drawHits(self):
        for idx, hit in enumerate(self.hits):
            if hit.distance > 0:
                s = str(idx) + " distance: " + str(hit.distance) + " (r:" + str(hit.radius) + " t: " + str(hit.time) + ")"
            else:
                s = str(idx) + " (r:" + str(hit.radius) + " t: " + str(hit.time) + ")"

            if idx == 0:
                color = (0, 200, 0)
            elif idx == 1:
                color = (0, 100, 240)
            elif idx == 2:
                color = (150, 0, 200)
            else:
                color = (0, 170, 200)

            cv2.putText(self.frame, s, (0, 0 + 140 + (30 * idx)), cv2.FONT_HERSHEY_TRIPLEX, 1.0, color, 2)
            self.pluginHits.draw(self.frame, hit)
            #cv2.circle(self.frame, (hit.x, hit.y), hit.radius, color, 2)
            #cv2.circle(self.frame, (hit.x, hit.y), 10, color, -1)


    def drawGameMode(self):
        if not self.gameMode.isShowStart():
            return

        color = (0, 100, 240)
        cv2.putText(self.frame, "Shoot!", (500, 440), cv2.FONT_HERSHEY_TRIPLEX, 1.0, color, 2)


    def saveCurrentFrame(self, recordedHit=None):
        """Save current frame as file"""
        filenameBase = self.videoStream.getFilenameBase()
        filenameBase += '_' + str(self.frameNr) + '_'

        logger.info("Saving current frame:")
        if recordedHit is not None:
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


    def setFrameRel(self, frameOffset):
        print("Lazer: Set Frame rel: {} @ {}".format(frameOffset, self.frameNr))
        self.detectorThread.setFrameNr(self.frameNr + frameOffset)


    def setCrop(self, crop):
        self.threadData['crop'] = crop


    #def addThresh(self, thresh):
    #    self.threadData['thresh'] += thresh


    def setThresh(self, thresh):
        self.threadData['thresh'] = thresh


    def getThresh(self):
        return self.threadData['thresh']


    def getMode(self):
        return self.threadData['mode']


    def release(self):
        self.detectorThread.shutdownThread()
