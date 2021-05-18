
import glob
import time
import yaml
import os.path
import cv2 as cv

from lazer import Lazer

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
    'test30',       # tested before, normal
    'test31',       # more wide view, but normal
    'test42',       # gopro 120fps
]


def getTime():
    return time.time()
    #returm time.process_time()
    #return time.clock()


def doTestsQuick():
    results = []

    for test in tests: 
        start = getTime()
        frameCnt = testcaseQuick(test)
        end = getTime()
        res = {
            'test': test,
            'time': end - start,
        }
        results.append(res)

    for result in results:
        print("Times: " + result['test'] + ": " + str( result['time'] ))


def testcaseQuick(filename, showVid=False): 
    print("Test file: " + filename)
    lazer = Lazer(showVid=showVid, showGlare=False, saveFrames=False, saveHits=False)
    
    # get all testcases to check if all triggered
    yamlFilenameList = glob.glob('tests/' + filename + "_*.yaml")
    yamlFilenameList = [i.replace('\\', '/') for i in yamlFilenameList]

    for yamlFile in yamlFilenameList:
        print("  YamlFile: " + yamlFile)
        lazer.initFile("tests/" + filename + ".mp4")

        frameNr = int(yamlFile.split('_')[-2])
        lazer.setFrame(frameNr)
        hasFrame = lazer.nextFrame()

        # find contours and visualize it in the main frame
        recordedHits = lazer.detectAndDrawHits(staticImage=True)
        if len(recordedHits) > 0:
            recordedHit = recordedHits[0]
            print("Checking dot in frame " + str(lazer.frameNr))

            yamlFilename = "tests/" + filename + "_" + str(lazer.frameNr) + '_info.yaml'
            if not os.path.isfile(yamlFilename):
                print("Error: dot detectd, but no tastecase for " + yamlFilename)
                return

            with open(yamlFilename) as file:
                yamlRecordedHit = yaml.load(file, Loader=yaml.FullLoader)

                if abs(recordedHit.x - yamlRecordedHit['x']) > 10:
                    print("Error in dot coordinates: ")
                    print("  recordedHit.x  : " + str(recordedHit.x))
                    print("  yamlRecordHit.x: " + str(yamlRecordedHit['x']))
                if abs(recordedHit.y - yamlRecordedHit['y']) > 10:
                    print("Error in dot coordinates: ")
                    print("  recordedHit.y  : " + str(recordedHit.y))
                    print("  yamlRecordHit.y: " + str(yamlRecordedHit['y']))
        else:
            print("No hits :(")

        lazer.release()

    return 0


def doTests():
    results = []

    for test in tests: 
        start = getTime()
        frameCnt = testcase(test)
        end = getTime()
        
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
    lazer = Lazer(showVid=False, showGlare=False, threaded=True)
    lazer.initFile("tests/" + filename + ".mp4")

    # get all testcases to check if all triggered
    yamlFilenameList = glob.glob('tests/' + filename + "_*.yaml")
    yamlFilenameList = [i.replace('\\', '/') for i in yamlFilenameList]

    while True:
        hasFrame = lazer.nextFrame()
        if not hasFrame:
            break

        # find contours and visualize it in the main frame
        recordedHits = lazer.detectAndDrawHits()
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
        recordedHits = lazer.detectAndDrawHits()
        for recordedHit in recordedHits:
            filenameBase = os.path.splitext(filename)[0]
            # write all the pics
            cv.imwrite(filenameBase + "_" + str(lazer.frameNr) + "_mask.jpg", lazer.mask)
            cv.imwrite(filenameBase + "_" + str(lazer.frameNr) + "_frame.jpg", lazer.frame)
            #cv.imwrite(filenameBase + "_" + str(lazer.frameNr) + "_diff.jpg", lazer.diff)
            with open(filenameBase + "_" + str(lazer.frameNr) + "_info.yaml", 'w') as outfile:
                yaml.dump(recordedHit.toDict(), outfile)

    lazer.release()

