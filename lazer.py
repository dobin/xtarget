import cv2 as cv
import imutils
from collections import deque
from skimage.metrics import structural_similarity as ssim
import os.path
import yaml

from gfxutils import *
from model import *

# from ball_tracking.py
def findContours(mask, minRadius=5):
    res = []

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv.contourArea)
        ((x, y), radius) = cv.minEnclosingCircle(c)
        M = cv.moments(c)
        if M["m00"] == 0:
            ##print("DIVISION BY ZERO")
            return res
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
        if radius > minRadius:  # orig: 10, for most: 5
            #print("Found dot with radius " + str(radius) + "at  X:" + str(x) + "  Y:" + str(y))

            recordedHit = RecordedHit()
            recordedHit.x = int(x)
            recordedHit.y = int(y)
            recordedHit.center = center
            recordedHit.radius = int(radius)
            res.append(recordedHit)
        else:
            #print("Too small: " + str(radius))
            pass

    return res

# Order:
# - init*
# - nextFrame()
# - detectAndDrawHits()
# - displayFrame()
# - release()
class Lazer(object):
    def __init__(self, showVid, crop=None, showGlare=True, saveFrames=False, saveHits=False, endless=False):
        self.capture = None
        self.frame = None
        self.mask = None
        self.diff = None
        self.previousMask = None
        self.frameNr = -1  # so it is 0 the first iteration
        self.lastFoundFrameNr = 0
        self.showVid = showVid
        self.showGlare = showGlare
        self.graceTime = 10  # How many frames between detections
        self.filename = None
        self.crop = crop
        self.width = None
        self.height = None

        # debug options
        self.saveFrames = saveFrames
        self.saveHits = saveHits

        # playback options
        self.endless = endless

        # decoding options
        self.minRadius = 5
        self.thresh = 14
        self.doDenoise = True
        self.doSharpen = True

        self.mode = Mode.intro
        self.centerX = 0
        self.centerY = 0
        self.hitRadius = 0

        # to delete
        self.q = deque(maxlen=3)
        self.enableDiff = False

        self.init()


    def init(self):
        self.glareMeter = 0
        self.glareMeterAvg = 0
        self.hits = []
        self.lastFoundFrameNr = 0


    def changeMode(self, mode):
        self.mode = mode
        if mode == Mode.main:
            self.showGlare = False
        elif mode == Mode.intro:
            self.showGlare = True


    def setFrame(self, frameNr):
        self.capture.set(cv.CAP_PROP_POS_FRAMES, frameNr)
        self.frameNr = frameNr-1


    def setCrop(self, crop):
        self.crop = crop


    def getDistanceToCenter(self, x, y):
        return calculateDistance(self.centerX, self.centerY, x, y)

    
    def setCenter(self, x, y, hitRadius):
        print("Set center: " + str(x) + " / " + str(y))
        self.centerX = int(x)
        self.centerY = int(y)
        self.hitRadius = int(hitRadius)


    def initFile(self, filename):
        if not os.path.isfile(filename):
            print("File not found")
            return

        self.filename = filename
        self.capture = cv.VideoCapture(filename)

        # check for crop settings for file
        vidYaml = filename +'.yaml'
        if os.path.isfile(vidYaml):
            print("Has croppings...")
            with open(vidYaml) as file:
                vidYamlData = yaml.load(file, Loader=yaml.FullLoader)
                if 'x1' in vidYamlData:
                    self.crop = []
                    self.crop.append((vidYamlData['x1'], vidYamlData['y1']))
                    self.crop.append((vidYamlData['x2'], vidYamlData['y2']))
                if 'thresh' in vidYamlData:
                    self.thresh = vidYamlData['thresh']

        self.width = int(self.capture.get(cv.CAP_PROP_FRAME_WIDTH ))
        self.height = int(self.capture.get(cv.CAP_PROP_FRAME_HEIGHT ))


    def nextFrame(self):
        self.frameNr += 1
        self.previousMask = self.mask

        isTrue, self.frame = self.capture.read()
        if not isTrue:  # end of file
            if self.endless:
                self.init()
                self.setFrame(0)  # seamlessly start at the beginning
                _, self.frame = self.capture.read()
            else:
                return False

        # Crop if user wants to
        if self.crop != None:
            x1 = self.crop[0][0]
            y1 = self.crop[0][1]
            x2 = self.crop[1][0]
            y2 = self.crop[1][1]
            self.frame = self.frame[y1:y2, x1:x2]

        # Mask: Make to grey
        self.mask = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)

        # Mask: make it a bit sharper
        if self.doSharpen:
            self.mask = cv.medianBlur(self.mask,5)
            self.mask = cv.erode(self.mask, (7,7), iterations=3)
        
        # Mask: threshold, throw away all bytes below thresh (binary)
        _, self.mask = cv.threshold(self.mask, 255-self.thresh, 255, cv.THRESH_BINARY)

        # Mask: Check if there is glare and handle it (glaremeter, drawing rectangles)
        if self.showGlare:
            self.checkGlare()

        # if we wanna record everything
        if self.saveFrames:
            self.saveCurrentFrame()

        return True


    def detectAndDrawHits(self, staticImage=False):
        if not staticImage:
            # wait a bit between detections
            if (self.frameNr - self.lastFoundFrameNr) < self.graceTime:
                return []

            # check if there is any change at all
            # if no change, do not attempt to find contours. 
            # this can save processing power
            if frameIdentical(self.mask, self.previousMask):
                return []

        recordedHits = findContours(self.mask, self.minRadius)
        if len(recordedHits) > 0:
            self.lastFoundFrameNr = self.frameNr
            print(" --> Found hit at frame #" + str(self.frameNr) + " with radius " + str(recordedHits[0].radius))
        else:
            return []

        # augment mask,frame,diff with indicators?
        if staticImage or self.mode == Mode.main: # and not self.showGlare:
            for recordedHit in recordedHits:
                cv.circle(self.frame, (int(recordedHit.x), int(recordedHit.y)), int(recordedHit.radius), (0, 100, 50), 2)
                cv.circle(self.frame, recordedHit.center, 5, (0, 250, 50), -1)

                if self.hitRadius > 0:
                    p = int(self.getDistanceToCenter(recordedHit.x, recordedHit.y))
                    r = self.hitRadius
                    d = int(p/r * 100)
                    recordedHit.distance = d

                #cv.circle(self.mask, (int(recordedHit.x), int(recordedHit.y)), int(recordedHit.radius), (0, 100, 50), 2)
                #cv.circle(self.mask, recordedHit.center, 5, (0, 250, 50), -1)

                #cv.circle(self.diff, (int(recordedHit.x), int(recordedHit.y)), int(recordedHit.radius), (0, 100, 50), 2)
                #cv.circle(self.diff, recordedHit.center, 5, (0, 250, 50), -1)
                self.hits.append(recordedHit)

                if self.saveHits:
                    self.saveCurrentFrame()

        return recordedHits


    def checkGlare(self):
        # threshold the difference image, followed by finding contours to
        # obtain the regions of the two input images that differ
        #thresh = cv.threshold(diff, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
        #cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        cnts = cv.findContours(self.mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # loop over the contours
        for c in cnts:
            # compute the bounding box of the contour and then draw the
            # bounding box on both input images to represent where the two
            # images differ
            (x, y, w, h) = cv.boundingRect(c)
            cv.rectangle(self.frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            #cv.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

        if len(cnts) > 0:
            if self.glareMeter < 60:  # 30 is typical fps, so 1s
                self.glareMeter += 4
        else:
            if self.glareMeter > 0:
                self.glareMeter -= 1
        self.glareMeterAvg = int(self.glareMeterAvg + self.glareMeter) >> 1


    def displayFrame(self):
        o = 300

        color = (255, 255, 255)
        s = 'Frame: '+ str(self.frameNr)
        cv.putText(self.frame, s, (0,30), cv.FONT_HERSHEY_TRIPLEX, 1.0, color, 2)
        s= "Tresh: " + str(self.thresh) # + "  Glare: " + str(self.glareMeterAvg)
        cv.putText(self.frame, s, (o*1,30), cv.FONT_HERSHEY_TRIPLEX, 1.0, color, 2)
        if self.glareMeterAvg > 0:
            cv.putText(self.frame, "Glare: " + str(self.glareMeterAvg), (0,140), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0, 0, 255), 2)

        s = "Denoise: " + str(self.doDenoise)
        cv.putText(self.frame, s, ((o*0),60), cv.FONT_HERSHEY_TRIPLEX, 1.0, color, 2)
        s= "Sharpen: " + str(self.doSharpen)
        cv.putText(self.frame, s, (o*1,60), cv.FONT_HERSHEY_TRIPLEX, 1.0, color, 2)        

        s = "Mode: " + str(self.mode.name)
        cv.putText(self.frame, s, (o*1,90), cv.FONT_HERSHEY_TRIPLEX, 1.0, color, 2)        

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

            cv.putText(self.frame, s, (0,0+140+(30*idx)), cv.FONT_HERSHEY_TRIPLEX, 1.0, color, 2)
            cv.circle(self.frame, (hit.x, hit.y), hit.radius, color, 2)
            cv.circle(self.frame, (hit.x, hit.y), 10, color, -1)

        color = (0, 0, 255)
        if self.mode == Mode.intro:
            s = "Press m to start"
            cv.putText(self.frame, s, (self.width >> 1,self.height - 30), cv.FONT_HERSHEY_TRIPLEX, 1.0, color, 2)        
        elif self.mode == Mode.main:
            s = "Press m to stop"
            cv.putText(self.frame, s, ((self.width >> 1) - 60,self.height - 30), cv.FONT_HERSHEY_TRIPLEX, 1.0, color, 2)        

        cv.imshow('Video', self.frame)
        cv.imshow('Mask', self.mask)
        if self.diff is not None:
            cv.imshow('Diff', self.diff)


    def saveCurrentFrame(self, epilog=''):
        fname = self.filename + "." + str(self.frameNr) + '.hit.frame' + epilog + '.jpg'
        print("Save Frame to: " + fname)
        cv.imwrite(fname, self.frame)

        fname = self.filename + "." + str(self.frameNr) + '.hit.mask' + epilog + '.jpg' 
        print("Save Mask to: " + fname)
        cv.imwrite(fname, self.mask)

        #cv.imwrite(self.filename + "." + str(self.frameNr) + '.diff.jpg' , self.diff)


    def release(self):
        self.capture.release()
        if self.showVid:
            cv.destroyAllWindows()
        print("")
