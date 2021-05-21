from math import e
import cv2 as cv
import os
import logging

from fps import Fps
from inputstream import SimpleInputStream, QueueInputStream
from model import CamConfig


class VideoStream(object):
    def __init__(self, threaded):
        self.threaded = threaded

        self.inputStream = None
        self.crop = None
        self.frameNr = -1  # so it is 0 the first iteration
        self.filename = None
        self.width = None
        self.height = None

        self.fps = Fps()
        if threaded:
            logging.info("Using threads")
        

    def getFilenameBase(self):
        return os.path.splitext(self.filename)[0]


    def getFrame(self):
        self.fps.tack()
        self.frameNr += 1

        isTrue, frame = self.inputStream.read()
        if isTrue and self.crop != None:
            frame = self.doCrop(frame)

        return isTrue, frame


    def setCrop(self, crop):
        self.crop = crop


    def doCrop(self, frame):
        if self.crop == None:
            return

        x1 = self.crop[0][0]
        y1 = self.crop[0][1]
        x2 = self.crop[1][0]
        y2 = self.crop[1][1]
        frame = frame[y1:y2, x1:x2]

        return frame


    def setFrame(self, frameNr):
        # not for live streams, e.g. webcam
        pass


    def release(self):
        self.inputStream.release()


class FileVideoStream(VideoStream):
    def __init__(self, threaded, endless):
        super().__init__(threaded)
        self.endless = endless


    def initFile(self, filename):
        if not os.path.isfile(filename):
            logging.error("File not found: " + filename)
            return False
        self.filename = filename

        if self.threaded:
            self.inputStream = QueueInputStream(filename)
            self.inputStream.initStream()
            self.inputStream.start()  # start the reader thread
        else:
            self.inputStream = SimpleInputStream(filename)
            self.inputStream.initStream()

        self.width = int(self.inputStream.capture.get(cv.CAP_PROP_FRAME_WIDTH ))
        self.height = int(self.inputStream.capture.get(cv.CAP_PROP_FRAME_HEIGHT ))

        return True


    def getFrame(self):
        isTrue, frame = super().getFrame()
        if not isTrue and self.endless:  # if file ends, continue at the beginning 
            self.setFrame(0)  # seamlessly start at the beginning
            return super().getFrame()

        return isTrue, frame


    def setFrame(self, frameNr):
        if self.threaded:  # not implemented for now
            return

        self.inputStream.capture.set(cv.CAP_PROP_POS_FRAMES, frameNr)
        self.frameNr = frameNr-1


class CamVideoStream(VideoStream):
    def __init__(self, threaded):
        super().__init__(threaded)
        c = CamConfig()
        self.exposure = c.exposure
        self.gain = c.gain
        self.autoExposure = c.autoExposure


    def initCam(self, camId, resolution):
        self.filename = "cam_" + str(camId) + ".mp4"
        print("Initialize the cam. This can take some time...")
        if self.threaded:
            self.inputStream = QueueInputStream(camId)
            self.inputStream.initStream()
            self.inputStream.capture.set(3,resolution['width'])
            self.inputStream.capture.set(4,resolution['height'])
            self.inputStream.start()  # start the reader thread
        else:
            self.inputStream = SimpleInputStream(camId)
            self.inputStream.initStream()

            self.inputStream.capture.set(3,resolution['width'])
            self.inputStream.capture.set(4,resolution['height'])

        #self.width = int(self.inputStream.capture.get(cv.CAP_PROP_FRAME_WIDTH ))
        #self.height = int(self.inputStream.capture.get(cv.CAP_PROP_FRAME_HEIGHT ))
        self.width = resolution['width']
        self.height = resolution['height']

        logging.debug("Camera settings (most likely wrong): ")
        logging.debug("  Exposure: " + str(self.inputStream.capture.get(cv.CAP_PROP_EXPOSURE)))
        logging.debug("  Gain    : " + str(self.inputStream.capture.get(cv.CAP_PROP_GAIN)))
        logging.debug("  AutoExpo: " + str(self.inputStream.capture.get(cv.CAP_PROP_AUTO_EXPOSURE)))


    def updateCamSettings(self, camConfig):
        if camConfig.thresh != self.thresh:
            logging.info("Update Thresh: " + str(camConfig.thresh))
            self.thresh = camConfig.thresh
            
        if camConfig.exposure != self.exposure:
            logging.info("Update exposure: " + str(camConfig.exposure))
            self.exposure = camConfig.exposure
            self.capture.set(cv.CAP_PROP_EXPOSURE, self.exposure)

        if camConfig.gain != self.gain:
            logging.info("Update gain: " + str(camConfig.gain))
            self.gain = camConfig.gain
            self.capture.set(cv.CAP_PROP_GAIN, self.gain)

        if camConfig.autoExposure != self.autoExposure:
            self.autoExposure = self.autoExposure
            self.capture.set(cv.CAP_PROP_AUTO_EXPOSURE, self.autoExposure)