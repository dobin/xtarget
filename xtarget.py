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
import glob


def doTests():
    testcase("test3")
    testcase("test11")
    testcase("test12")
    testcase("test13")
    testcase("test20-far")
    testcase("test21")
    testcase("test22")


def testcase(filename):
    print("Test file: " + filename)
    lazer = Lazer(showVid=False)
    lazer.initFile("tests/" + filename + ".mp4")

    # get all testcases to check if all triggered
    yamlFilenameList = glob.glob('tests/' + filename + "_*.yaml")
    yamlFilenameList = [i.replace('\\', '/') for i in yamlFilenameList]

    while True:
        hasFrame = lazer.nextFrame()
        if not hasFrame:
            break

        # find contours and visualize it in the main frame
        recordedHits = lazer.getContours()
        if len(recordedHits) > 0:
            recordedHit = recordedHits[0]
            print("Checking dot in frame " + str(lazer.frameNr))

            yamlFilename = "tests/" + filename + "_" + str(lazer.frameNr) + '_info.yaml'
            if yamlFilename in yamlFilenameList:
                yamlFilenameList.remove(yamlFilename)
            else:
                print("Err: Found dot with no testcase at frame " + str(lazer.frameNr))
                continue

            #if not os.path.isfile(yamlFilename):
            #    print("Error: dot detectd, but no tastecase for " + yamlFilename)
            #    return

            with open(yamlFilename) as file:
                yamlRecordedHit = yaml.load(file, Loader=yaml.FullLoader)

                if abs(recordedHit.x - yamlRecordedHit['x']) > 10:
                    print("Error in dot coordinates: ")
                    print("  recordedHit.x  : " + str(recordedHit.x))
                    print("  yamlRecordHit.x: " + str(yamlRecordedHit['x']))
                #else:
                #    print("  X OK")
                if abs(recordedHit.y - yamlRecordedHit['y']) > 10:
                    print("Error in dot coordinates: ")
                    print("  recordedHit.y  : " + str(recordedHit.y))
                    print("  yamlRecordHit.y: " + str(yamlRecordedHit['y']))
                #else:
                #    print("  Y OK")

    if len(yamlFilenameList) != 0:
        print("Error: Following dots were not detected: " + str(yamlFilenameList))

    lazer.release()


def writeVideoInfo(filename):
    print("Analyzing file: " + filename)
    lazer = Lazer(showVid=False)
    lazer.initFile(filename)

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

    cv.namedWindow('Video')
    while True:
        hasFrame = lazer.nextFrame()
        if not hasFrame:
            lazer.capture.set(cv.CAP_PROP_POS_FRAMES, 0)
            continue

        # find contours and visualize it in the main frame
        contours = lazer.getContours()
        lazer.displayFrame()

        if False:
            cv.imwrite('test20-far.' + str(lazer.frameNr) + '.frame.jpg' , lazer.frame)

        # input
        key = cv.waitKey(2)
        if key == ord('c'):  # Crop image
            cv.setMouseCallback('Video', lazer.extract_coordinates_callback)
            while True:
                key = cv.waitKey(2)
                lazer.displayFrame()
                #cv.imshow('image', staticROI.clone)

                if key == ord('c'):
                    break
        if key == ord('q'):
            break

    lazer.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="video", action='store_true')
    ap.add_argument("-t", "--test", help="test", action='store_true')
    ap.add_argument("-w", "--write", help="write", action='store_true')
    ap.add_argument("-f", "--file", help="file", type=str)
    args = ap.parse_args()

    filename = args.file
    if args.video:
        analyzeVideo(filename)
    elif args.test:
        doTests()
    elif args.write:
        writeVideoInfo(filename)

writeVideoInfo

if __name__ == "__main__":
    main()