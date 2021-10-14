import logging

from model import *

logger = logging.getLogger(__name__)


class GameMode(object):

    def __init__(self):
        self.frameNrStart = 0
        self.frameNr = 0


    def isShowStart(self):
        if self.frameNrStart == 0:
            return

        if self.frameNr > self.frameNrStart and self.frameNr < self.frameNrStart + 15:
            return True
        else:
            return False

    def start(self):
        logger.info("Start")
        self.frameNrStart = self.frameNr + 90 # around 3s

    def reset(self):
        ret = self.frameNr-self.frameNrStart
        self.start()
        return ret

    def stop(self):
        pass

    def nextFrame(self, frameNr):
        self.frameNr = frameNr

