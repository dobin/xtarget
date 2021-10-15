import logging
import cv2

logger = logging.getLogger(__name__)


class PluginGlare(object):
    def __init__(self):
        self.init()


    def init(self):
        self.glareMeter = 0
        self.glareMeterAvg = 0


    def handle(self, frame, glare):
        if len(glare) > 0:
            if self.glareMeter < 60:  # 30 is typical fps, so 1s
                self.glareMeter += 4
        else:
            if self.glareMeter > 0:
                self.glareMeter -= 1
        self.glareMeterAvg = int(self.glareMeterAvg + self.glareMeter) >> 1

        self.draw(frame, glare)


    def draw(self, frame, glare):
        for rect in glare:
            cv2.rectangle(frame, (rect.x, rect.y), (rect.x + rect.w, rect.y + rect.h), (0, 0, 255), 2)
