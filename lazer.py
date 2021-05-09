import cv2 as cv
import imutils
from collections import deque
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os.path
from utils import *
from pathlib import Path
import yaml
import argparse


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
def findContours(mask):
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
        if radius > 10:
            #print("Found dot with radius " + str(radius) + "at  X:" + str(x) + "  Y:" + str(y))

            recordedHit = RecordedHit()
            recordedHit.x = x
            recordedHit.y = y
            recordedHit.center = center
            recordedHit.radius = radius
            res.append(recordedHit)

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


# Order:
# - init*
# - nextFrame()
# - getContours()
# - displayFrame()
# - release()
class Lazer(object):
    def __init__(self, showVid, showGlare=True):
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

        # for extract_coordinates_callback
        self.image_coordinates = None
        self.selected_ROI = False


    def initFile(self, filename):
        if not os.path.isfile(filename):
            print("File not found")
            return

        self.capture = cv.VideoCapture(filename)

        # check for crop settings for file
        vidYaml = filename +'.yaml'
        if os.path.isfile(vidYaml):
            print("Has croppings...")
            with open(vidYaml) as file:
                vidYamlData = yaml.load(file, Loader=yaml.FullLoader)
                self.image_coordinates = []
                self.image_coordinates.append((vidYamlData['x1'], vidYamlData['y1']))
                self.image_coordinates.append((vidYamlData['x2'], vidYamlData['y2']))
                self.selected_ROI = True


    def nextFrame(self):
        self.frameNr += 1
        self.previousMask = self.mask

        isTrue, self.frame = self.capture.read()
        if not isTrue:
            return False

        # Frame: rescale it
        self.frame = rescaleFrame(self.frame)

        # Crop if user wants to
        if self.selected_ROI:
            x1 = self.image_coordinates[0][0]
            y1 = self.image_coordinates[0][1]
            x2 = self.image_coordinates[1][0]
            y2 = self.image_coordinates[1][1]
            self.frame = self.frame[y1:y2, x1:x2]

        # we have three posibilities to make a mask: 
        # - cv.addedWeight brightness/constrast, like in gimp
        # - cv.inRange in hsv range to filter
        # - cv.threshhold RGB byte filter

        ###self.mask = filterShit(self.frame)

        # Mask: grey
        self.mask = toGrey(self.frame)

        #self.mask = sharpoon(self.mask)
        self.mask = trasholding(self.mask)

        # Mask: make it a bit sharper
        
        # Mask: force super low brightness high contrast
        #self.mask = apply_brightness_contrast(self.mask, -126, 115)
        #orig: mask = apply_brightness_contrast(mask, -127, 116)
        
        if self.showGlare:
            self.checkGlare()

        return True


    def getContours(self, addIndicator=True):
        # wait a bit between detections
        if (self.frameNr - self.lastFoundFrameNr) < self.graceTime:
            return []

        # check if there is any change at all
        # if no change, do not attempt to find contours. 
        # this can save processing power
        if frameIdentical(self.mask, self.previousMask):
            return []

        recordedHits = findContours(self.mask)
        if len(recordedHits) > 0:
            self.lastFoundFrameNr = self.frameNr
            print("--> Found hit")
        else:
            return []

        # also calulate the diff
        if self.mask.shape == self.previousMask.shape:
            self.diff = diff(self.mask, self.previousMask, self.frame)
        
        # augment mask,frame,diff with indicators?
        if addIndicator: # and not self.showGlare:
            for recordedHit in recordedHits:
                # add visual indicators to both frame and mask
                cv.circle(self.frame, (int(recordedHit.x), int(recordedHit.y)), int(recordedHit.radius), (255, 0, 0), 2)
                cv.circle(self.frame, recordedHit.center, 5, (0, 255, 0), -1)

                cv.circle(self.mask, (int(recordedHit.x), int(recordedHit.y)), int(recordedHit.radius), (255, 0, 0), 2)
                cv.circle(self.mask, recordedHit.center, 5, (0, 255, 0), -1)

                cv.circle(self.diff, (int(recordedHit.x), int(recordedHit.y)), int(recordedHit.radius), (255, 0, 0), 2)
                cv.circle(self.diff, recordedHit.center, 5, (0, 255, 0), -1)

                if False:
                    cv.imwrite("xxx_" + str(self.frameNr) + "frame.jpg", self.frame)
                    cv.imwrite("xxx_" + str(self.frameNr) + "mask.jpg", self.mask)
                    cv.imwrite("xxx_" + str(self.frameNr) + "diff.jpg", self.diff)

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

    
    def extract_coordinates_callback(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv.EVENT_LBUTTONDOWN:
            self.image_coordinates = [(x,y)]

        # Record ending (x,y) coordintes on left mouse bottom release
        elif event == cv.EVENT_LBUTTONUP:
            self.image_coordinates.append((x,y))
            self.selected_ROI = True

            # Draw rectangle around ROI
            cv.rectangle(self.frame, self.image_coordinates[0], self.image_coordinates[1], (0,255,0), 2)
            print("ROI: " + str(self.image_coordinates))

        # Clear drawing boxes on right mouse button click
        elif event == cv.EVENT_RBUTTONDOWN:
            self.selected_ROI = False


    def displayFrame(self):
        cv.imshow('Video', self.frame)
        cv.imshow('Mask', self.mask)
        if self.diff is not None:
            cv.imshow('Diff', self.diff)


    def release(self):
        self.capture.release()
        if self.showVid:
            cv.destroyAllWindows()
