from detector import Detector
from threading import Thread
from queue import Queue

from model import *


class DetectorThread():
    def __init__(self, videoStream):
        self.videoStream = videoStream
        self.stopped = False
        self.detector = Detector(videoStream)
        self.Q = Queue(maxsize=8)


    def startThread(self, modeShared):
        self.thread = Thread(target=self.update, args=(modeShared, ))
        self.thread.daemon = True
        self.thread.start()


    def shutdownThread(self):
        self.stopped = True
        self.videoStream.release()
        self.thread.join()

    
    def update(self, threadData):
        while not self.stopped:
            isTrue, frame, frameNr = self.videoStream.getFrame()
            if not isTrue:
                self.stopped = True
                self.Q.put((False, None, frameNr, None, None))
                break

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
                self.Q.put((isTrue, frame, frameNr, Mode.intro, data))

            elif threadData['mode'] == Mode.main:
                recordedHits = self.detector.findHits(minRadius=1.0)
                data = {
                    'recordedHits': recordedHits,
                }
                self.Q.put((isTrue, frame, frameNr, Mode.main, data))

