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
from lazer import RecordedHit, Lazer


def doTests():
    testcase("test2")
    testcase("test3")


def testcase(filename):
    print("Test file: " + filename)
    lazer = Lazer(showVid=False)
    lazer.initFile("tests/" + filename + ".mp4")

    while True:
        hasFrame = lazer.nextFrame()
        if not hasFrame:
            break

        # find contours and visualize it in the main frame
        contours = lazer.getContours()
        if len(contours) > 0:
            print(lazer.frameNr)

    lazer.release()


def writeVideoInfo(filename):
    print("Analyzing file: " + filename)
    lazer = Lazer(showVid=False)
    lazer.initFile("tests/" + filename + ".mp4")

    while True:
        hasFrame = lazer.nextFrame()
        if not hasFrame:
            break

        # find contours and visualize it in the main frame
        recordedHits = lazer.getContours()
        for recordedHit in recordedHits:
            filenameBase = os.path.splitext(filename)[0]
            # write all the pics
            cv.imwrite(filenameBase + "_" + str(lazer.frameNr) + "_cont.jpg", lazer.mask)
            cv.imwrite(filenameBase + "_" + str(lazer.frameNr) + "_frame.jpg", lazer.frame)
            cv.imwrite(filenameBase + "_" + str(lazer.frameNr) + "_diff.jpg", lazer.diff)
            with open(filenameBase + "_" + str(lazer.frameNr) + "_info.yaml", 'w') as outfile:
                yaml.dump(recordedHit.toDict(), outfile)

    lazer.release()


def analyzeVideo(filename):
    print("Analyzing file: " + filename)
    lazer = Lazer(showVid=True)
    lazer.initFile(filename)

    while True:
        hasFrame = lazer.nextFrame()
        if not hasFrame:
            break

        # find contours and visualize it in the main frame
        contours = lazer.getContours()
        lazer.displayFrame()

        # check for end
        if cv.waitKey(20) & 0xFF==ord('d'):
            break

    lazer.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="video", action='store_true')
    ap.add_argument("-t", "--test", help="test", action='store_true')
    args = ap.parse_args()

    if args.video:
        analyzeVideo('tests/test2.mp4')
    elif args.test:
        doTests()


if __name__ == "__main__":
    main()