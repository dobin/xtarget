from math import e
import yaml
import cv2 as cv
import os

from fps import Fps
from inputstream import SimpleInputStream, QueueInputStream


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
        print("Threaded: " + str(threaded))
        

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


    def release(self):
        pass


class FileVideoStream(VideoStream):
    def __init__(self, threaded, endless):
        super().__init__(threaded)
        self.endless = endless


    def initFile(self, filename):
        if not os.path.isfile(filename):
            print("File not found")
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


    def release(self):
        self.inputStream.release()


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
        self.thresh = c.thresh
        self.exposure = c.exposure
        self.gain = c.gain
        self.autoExposure = c.autoExposure


    def initCam(self, camId):
        self.filename = "cam_" + str(camId)
        print("Threaded: " + str(self.threaded))
        if self.threaded:
            self.capture = QueueVideoStream(camId).start()
            #time.sleep(1)  # give cam time to autofocus etc. 
            self.width = int(self.capture.stream.get(cv.CAP_PROP_FRAME_WIDTH ))
            self.height = int(self.capture.stream.get(cv.CAP_PROP_FRAME_HEIGHT ))
            print("Exposure: " + str(self.capture.stream.get(cv.CAP_PROP_EXPOSURE)))
            print("gain: " + str(self.capture.stream.get(cv.CAP_PROP_GAIN)))
        else:
            self.capture = cv.VideoCapture(camId)
            self.width = int(self.capture.get(cv.CAP_PROP_FRAME_WIDTH ))
            self.height = int(self.capture.get(cv.CAP_PROP_FRAME_HEIGHT ))
            print("Exposure: " + str(self.capture.get(cv.CAP_PROP_EXPOSURE)))
            print("gain: " + str(self.capture.get(cv.CAP_PROP_GAIN)))
            print("AutoExpo: " + str(self.capture.get(cv.CAP_PROP_AUTO_EXPOSURE)))

            # hardcode resolution for now
            self.capture.set(3,1920)
            self.capture.set(4,1080)



    def updateCamSettings(self, camConfig):
        if camConfig.thresh != self.thresh:
            print("Update Thresh: " + str(camConfig.thresh))
            self.thresh = camConfig.thresh
            
        if camConfig.exposure != self.exposure:
            print("Update exposure: " + str(camConfig.exposure))
            self.exposure = camConfig.exposure
            self.capture.set(cv.CAP_PROP_EXPOSURE, self.exposure)

        if camConfig.gain != self.gain:
            print("Update gain: " + str(camConfig.gain))
            self.gain = camConfig.gain
            self.capture.set(cv.CAP_PROP_GAIN, self.gain)

        if camConfig.autoExposure != self.autoExposure:
            self.autoExposure = self.autoExposure
            self.capture.set(cv.CAP_PROP_AUTO_EXPOSURE, self.autoExposure)