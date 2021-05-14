import cv2 as cv
import imutils
from collections import deque
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os.path
from utils import *
from pathlib import Path
import yaml
from enum import Enum


class RecordedHit(object):
    def __init__(self):
        self.center = None
        self.x = 0
        self.y = 0
        self.radius = 0

    def toDict(self):
        me = {
            'center': self.center,
            'x': self.x,
            'y': self.y,
            'radius': self.radius
        }
        return me


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


def diff(mask, previousMask, frame):
    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    (score, diff) = ssim(mask, previousMask, full=True)
    diff = (diff * 255).astype("uint8")
    #print("SSIM: {}".format(score))

    if False:
        # threshold the difference image, followed by finding contours to
        # obtain the regions of the two input images that differ
        thresh = cv.threshold(diff, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
        cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # loop over the contours
        for c in cnts:
            # compute the bounding box of the contour and then draw the
            # bounding box on both input images to represent where the two
            # images differ
            (x, y, w, h) = cv.boundingRect(c)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            #cv.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

    #cv.imshow('diff', diff)
    return diff



class Mode(Enum):
    intro = 1
    main = 2


# Order:
# - init*
# - nextFrame()
# - getContours()
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


    def multiframeDenoise(self, mask):
        self.q.appendleft(mask)
        #print("Len: " + str(len(self.q)))

        x = 3
        if x == 1:
            # use previous frame to average it
            if len(self.q) != 3:
                return mask

            q = [
                self.q[1],
                self.q[0],
                self.q[1]
            ]
            mask = cv.fastNlMeansDenoisingMulti(q, 1, 1)


        if x == 2:
            # only the frame itself
            mask = cv.fastNlMeansDenoising(mask)

        if x == 3:
            pass

        return mask


    def setFrame(self, frameNr):
        self.capture.set(cv.CAP_PROP_POS_FRAMES, frameNr)
        self.frameNr = frameNr-1


    def nextFrame(self):
        self.frameNr += 1
        self.previousMask = self.mask

        isTrue, self.frame = self.capture.read()
        if not isTrue:
            if self.endless:
                self.init()
                self.hits = []
                self.setFrame(0)
                isTrue, self.frame = self.capture.read()
                
            else:
                return False

        # Frame: rescale it
        self.frame = rescaleFrame(self.frame)

        # Crop if user wants to
        if self.crop != None:
            x1 = self.crop[0][0]
            y1 = self.crop[0][1]
            x2 = self.crop[1][0]
            y2 = self.crop[1][1]
            self.frame = self.frame[y1:y2, x1:x2]

        # we have three posibilities to make a mask: 
        # - cv.addedWeight brightness/constrast, like in gimp
        # - cv.inRange in hsv range to filter
        # - cv.threshhold RGB byte filter

        ###self.mask = filterShit(self.frame)

        # Mask: grey
        self.mask = toGrey(self.frame)
        if self.doDenoise:
            self.mask = self.multiframeDenoise(self.mask)

        # Mask: make it a bit sharper
        #self.mask = sharpoon(self.mask)
        if self.doSharpen:
            self.mask = self.sharpen(self.mask)
        
        #self.mask = trasholding(self.mask)
        self.mask = self.masking(self.mask)
        
        # Mask: force super low brightness high contrast
        #self.mask = apply_brightness_contrast(self.mask, -126, 115)
        #orig: mask = apply_brightness_contrast(mask, -127, 116)
        
        # also calulate the diff
        if self.enableDiff and self.previousMask is not None:
            if self.mask.shape == self.previousMask.shape:
                self.diff = diff(self.mask, self.previousMask, self.frame)

        if self.saveFrames:
            self.saveCurrentFrame()

        if self.showGlare:
            self.checkGlare()

        return True


    def sharpen(self, frame):
        # Guassian Blur
        #frame = cv.GaussianBlur(frame, (3,3), cv.BORDER_DEFAULT)

        # median
        # TEST: Works for most
        frame = cv.medianBlur(frame,5)

        # simple
        # TEST: 
        #frame = cv.blur(frame,(5,5))

        # Dilating / blur=
        #frame = cv.dilate(frame, (7,7), iterations=3)

        # Eroding / shapren?
        frame = cv.erode(frame, (7,7), iterations=3)

        return frame


    def masking(self, mask):
        ret,thresh1 = cv.threshold(mask, 255-self.thresh, 255, cv.THRESH_BINARY)
        return thresh1


    def saveCurrentFrame(self):
        cv.imwrite(self.filename + "." + str(self.frameNr) + '.hit.frame.jpg' , self.frame)
        cv.imwrite(self.filename + "." + str(self.frameNr) + '.hit.mask.jpg' , self.mask)
        #cv.imwrite(self.filename + "." + str(self.frameNr) + '.diff.jpg' , self.diff)


    def getContours(self, staticImage=False):
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
            s = str(idx) + ": " + str(hit.x) + "/" + str(hit.y) + " r:" + str(hit.radius)
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


    def release(self):
        self.capture.release()
        if self.showVid:
            cv.destroyAllWindows()
        print("")
