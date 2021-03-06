from enum import Enum

class RecordedHit(object):
    def __init__(self):
        self.center = None
        self.x = 0
        self.y = 0
        self.radius = 0

        self.distance = 0
        self.time = 0

    def toDict(self):
        me = {
            'center': self.center,
            'x': self.x,
            'y': self.y,
            'radius': self.radius,
            'time': self.time,
        }
        return me


class CamConfig(object):
    def __init__(self):
        self.exposure = -7.0
        self.gain = 500.0
        self.autoExposure = 1.0


class Mode(Enum):
    intro = 1
    main = 2


class OpencvRect():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


class Relief():
    def __init__(self): 
        self.x = None
        self.y = None
        self.w = None
        self.h = None

        self.centerX = None
        self.centerY = None
