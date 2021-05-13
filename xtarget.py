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
import time

def doTests():
    # all recorded with surface book front camera 30fps if not stated otherwise
    tests = [
        'test3',        # room 1
        'test4_floor',  # room 1 on the floor with reflections (glare)
        'test11',       # room 2
        'test12',       # room 3
        'test13',       # room 4
        'test20-far',   # far away, sunny
        'test21',       # far away, dark
        'test22',       # oneplus 8 pro phone cam, 60fps, far, dark
    ]
    results = []

    for test in tests: 
        start = time.process_time()
        frameCnt = testcase(test)
        end = time.process_time()
        res = {
            'test': test,
            'time': end - start,
            'frames': frameCnt
        }
        results.append(res)

    for result in results:
        print("FPS: " + result['test'] + ": " + str( int(result['frames'] / result['time']) ))


def testcase(filename):
    print("Test file: " + filename)
    lazer = Lazer(showVid=False, showGlare=False)
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
    return lazer.frameNr


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
            cv.imwrite(filenameBase + "_" + str(lazer.frameNr) + "_mask.jpg", lazer.mask)
            cv.imwrite(filenameBase + "_" + str(lazer.frameNr) + "_frame.jpg", lazer.frame)
            #cv.imwrite(filenameBase + "_" + str(lazer.frameNr) + "_diff.jpg", lazer.diff)
            with open(filenameBase + "_" + str(lazer.frameNr) + "_info.yaml", 'w') as outfile:
                yaml.dump(recordedHit.toDict(), outfile)

    lazer.release()


def analyzeVideo(filename, saveFrames=False, saveHits=False):
    print("Analyzing file: " + filename)
    lazer = Lazer(showVid=True, saveFrames=saveFrames, saveHits=saveHits, endless=True)
    lazer.initFile(filename)

    cv.namedWindow('Video')
    while True:
        hasFrame = lazer.nextFrame()
        if not hasFrame:
            break

        # find contours and visualize it in the main frame
        contours = lazer.getContours()
        lazer.displayFrame()

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
    print("")


def showFrame(filename, frameNr):
    print("Analyzing file: " + filename)
    lazer = Lazer(showVid=True, showGlare=False)
    lazer.initFile(filename)

    cv.namedWindow('Video')  
    lazer.capture.set(cv.CAP_PROP_POS_FRAMES, frameNr)
    hasFrame = lazer.nextFrame()

    contours = lazer.getContours(staticImage=True)
    lazer.displayFrame()

    key = cv.waitKey(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="video", action='store_true')
    ap.add_argument("-t", "--test", help="test", action='store_true')
    ap.add_argument("-w", "--write", help="write", action='store_true')
    ap.add_argument("-s", "--showframe", help="showframe", action='store_true')

    ap.add_argument("-f", "--file", help="file", type=str)
    ap.add_argument("-n", "--nr", help="frame nr", type=int)

    ap.add_argument("--saveHits", action='store_true', default=False)
    ap.add_argument("--saveFrames", action='store_true', default=False)

    args = ap.parse_args()

    filename = args.file
    if args.video:
        analyzeVideo(filename, saveFrames=args.saveFrames, saveHits=args.saveHits)
    elif args.test:
        doTests()
    elif args.write:
        writeVideoInfo(filename)
    elif args.showframe:
        showFrame(filename, args.nr)

writeVideoInfo

if __name__ == "__main__":
    main()