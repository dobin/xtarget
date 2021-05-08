import cv2 as cv
import imutils
from collections import deque
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os.path
from utils import *
from pathlib import Path
import yaml

debug = True


class RecordedHit(object):
    def __init__(self):
        self.center = None
        self.x = 0
        self.y = 0
        self.radius = 0
        self.mask = None

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
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
        if radius > 10:
            print("Found dot with radius " + str(radius) + "at  X:" + str(x) + "  Y:" + str(y))

            recordedHit = RecordedHit()
            recordedHit.x = x
            recordedHit.y = y
            recordedHit.center = center
            recordedHit.radius = radius
            recordedHit.mask = mask.copy()
            res.append(recordedHit)

    return res


def diff(mask, previousMask, frame):
    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    (score, diff) = ssim(mask, previousMask, full=True)
    diff = (diff * 255).astype("uint8")
    #print("SSIM: {}".format(score))

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
        cv.rectangle(diff, (x, y), (x + w, y + h), (0, 0, 255), 2)
        #cv.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

    #cv.imshow('diff', diff)
    return diff


def analyzeVideo(filename, showVid=True):
    print("Analyzing file: " + filename)
    if not os.path.isfile(filename):
        print("File not found")
        return
    #filenameBase = Path(filename).stem
    filenameBase = os.path.splitext(filename)[0]


    # Reading Videos
    capture = cv.VideoCapture(filename)

    previousMask = None
    frameNr = 0
    lastFoundFrameNr = 0
    while True:
        isTrue, frame = capture.read()

        # rescale it        
        frame = rescaleFrame(frame)
        # grey
        mask = toGrey(frame)
        # make it a bit sharper
        mask = sharpoon(mask)
        # force super low brightness high contrast
        mask = apply_brightness_contrast(mask, -126, 115)
        # mask = apply_brightness_contrast(mask, -127, 116)
        if debug and showVid:
            cv.imshow('Mask', mask)

        # find movement / diffs
        #if previousMask is not None:
        #    # diff works well on brightness adjusted frame
        #    d = diff(mask, previousMask, frame)
        #    (frame, didFind) = findContours(d, frame, frameNr)
        #    if didFind:
        #        cv.imwrite(str(frameNr) + "_diff.jpg", d)

        # find contours and visualize it in the main frame
        if (frameNr - lastFoundFrameNr) > 10:
            recordedHits = findContours(mask)

            for recordedHit in recordedHits:
                lastFoundFrameNr = frameNr

                # also check the diff
                d = diff(mask, previousMask, frame)
                if showVid:
                    cv.imshow('Diff', d)
                
                # add visual indicators
                cv.circle(frame, (int(recordedHit.x), int(recordedHit.y)), int(recordedHit.radius), (255, 0, 0), 2)
                cv.circle(frame, recordedHit.center, 5, (0, 0, 0), -1)
                cv.circle(mask, (int(recordedHit.x), int(recordedHit.y)), int(recordedHit.radius), (255, 0, 0), 2)
                cv.circle(mask, recordedHit.center, 5, (0, 0, 0), -1)

                # write all the pics
                cv.imwrite(filenameBase + "_" + str(frameNr) + "_cont.jpg", recordedHit.mask)
                cv.imwrite(filenameBase + "_" + str(frameNr) + "_frame.jpg", frame)
                cv.imwrite(filenameBase + "_" + str(frameNr) + "_diff.jpg", d)
                with open(filenameBase + "_" + str(frameNr) + "_info.yaml", 'w') as outfile:
                    yaml.dump(recordedHit.toDict(), outfile)

        # show it
        if showVid:
            cv.imshow('Video', frame)

            # check for end
            if cv.waitKey(20) & 0xFF==ord('d'):
                break

        frameNr+=1
        previousMask = mask

    capture.release()
    cv.destroyAllWindows()


def main():
    analyzeVideo('tests/test1.mp4', showVid=False)

if __name__ == "__main__":
    main()