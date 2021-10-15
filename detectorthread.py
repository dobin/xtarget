import logging

from detector import Detector
from threading import Thread
from queue import Queue

from model import Mode

logger = logging.getLogger(__name__)


class DetectorThread():
    def __init__(self, videoStream):
        self.videoStream = videoStream
        self.detector = Detector(videoStream)

        # We support non-threaded implementation for file based video playback
        # as configured in the videostream
        self.doThread = videoStream.threaded

        # Thread stuff
        self.stopped = False
        self.Q = Queue(maxsize=8)
        self.threadData = None  # All configurable data consumed by this thread


    def startThread(self, threadData):
        if not self.doThread:
            self.threadData = threadData  # only need this if !doThread, funnily
            return
        self.thread = Thread(target=self.update, args=(threadData, ))  # give reference to threaddata to the thread function
        self.thread.daemon = True
        self.thread.start()


    def shutdownThread(self):
        if not self.doThread:
            return
        self.stopped = True
        self.videoStream.release()
        self.thread.join()


    def setFrameNr(self, frameNr):
        if self.doThread:
            logger.warn("SetFrameNr on threaded detector?")
            return
        self.videoStream.setFrame(frameNr)


    def getFrameData(self):
        if self.doThread:
            return self.Q.get()
        else:
            return self._getFrame(self.threadData)


    def update(self, threadData):
        """Thread: Main endless loop"""
        while not self.stopped:
            data = self._getFrame(threadData)
            self.Q.put(data)


    def _getFrame(self, threadData):
        isTrue, frame, frameNr = self.videoStream.getFrame()
        if not isTrue:
            self.stopped = True
            return((False, None, frameNr, None, None))

        self.detector.initFrame(frame, threadData['thresh'])
        if threadData['mode'] == Mode.intro:
            glare = self.detector.findGlare()
            contours, reliefs = self.detector.findTargets(threadData['targetThresh'])
            (corners, ids, rejected) = self.detector.findAruco()
            data = {
                'glare': glare,

                'targetContours': contours,
                'targetReliefs': reliefs,

                'arucoCorners': corners,
                'arucoIds': ids,
                'arucoRejected': rejected,
            }
            return((isTrue, frame, frameNr, Mode.intro, data))

        elif threadData['mode'] == Mode.main:
            recordedHits = self.detector.findHits(minRadius=1.0)
            data = {
                'recordedHits': recordedHits,
            }
            return((isTrue, frame, frameNr, Mode.main, data))
