from gfxutils import getTime
import time

class Fps(object):
    def __init__(self):
        self.ticklist = [0] * 100
        self.ticklistSize = 100
        self.tickIndex = 0
        self.ticksum = 0
        self.prevTime = 0

    def tick(self, newtick):
        self.ticksum -= self.ticklist[self.tickIndex]
        self.ticksum += newtick
        self.ticklist[self.tickIndex] = newtick

        self.tickIndex = (self.tickIndex + 1) % self.ticklistSize

    def tack(self):
        t = getTime()
        if self.prevTime > 0:
            d = t - self.prevTime
            self.tick(d)
        self.prevTime = t

    def get(self):
        avg = self.ticksum / self.ticklistSize
        if avg == 0:
            return 0
        else:
            return int(1/avg)

