
import glob
import time
import yaml
import os.path
import cv2 as cv

from gfxutils import getTime, readVideoFileConfig
from lazer import Lazer
from videostream import FileVideoStream


tests = [
    # surface book
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

    # gopro hero4
    'test42_out',       # 120fps
    
    # logitech c920
    # 'test52', # its .mkv
    'test60_itarget',
    'test62_paper',
    'test63_camfar',
    'test64_camfarer'
]


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


def testcaseQuick(basename):
    print("Test file: " + basename)
    filename = "tests/" + basename + ".mp4"

    videoFileConfig = readVideoFileConfig(filename)

    videoStream = FileVideoStream(threaded=False, endless=False)
    videoStream.setCrop(videoFileConfig['crop'])

    lazer = Lazer(videoStream, thresh=videoFileConfig['thresh'], saveFrames=False, saveHits=False)
    
    # get all testcases to check if all triggered
    yamlFilenameList = glob.glob('tests/' + basename + "_*.yaml")
    yamlFilenameList = [i.replace('\\', '/') for i in yamlFilenameList]

    for yamlFile in yamlFilenameList:
        print("  YamlFile: " + yamlFile)
        videoStream.initFile(filename)

        frameNr = int(yamlFile.split('_')[-2])
        videoStream.setFrame(frameNr)
        lazer.nextFrame()

        recordedHits = lazer.detectAndDrawHits(staticImage=True)
        if len(recordedHits) > 0:
            recordedHit = recordedHits[0]
            print("Checking dot in frame " + str(videoStream.frameNr))
            testHandleHit(recordedHit, basename, videoStream.frameNr, yamlFilenameList)
        else:
            print("Err: No hits detected :( ")
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


def testcase(basename):
    print("Test file: " + basename)
    filename = "tests/" + basename + ".mp4"

    videoFileConfig = readVideoFileConfig(filename)

    videoStream = FileVideoStream(threaded=True, endless=False)
    videoStream.initFile(filename)
    videoStream.setCrop(videoFileConfig['crop'])

    lazer = Lazer(videoStream, thresh=videoFileConfig['thresh'], saveFrames=False, saveHits=False)

    # get all testcases to check if all triggered
    yamlFilenameList = glob.glob('tests/' + basename + "_*.yaml")
    yamlFilenameList = [i.replace('\\', '/') for i in yamlFilenameList]

    while True:
        hasFrame = lazer.nextFrame()
        if not hasFrame:
            break

        recordedHits = lazer.detectAndDrawHits()
        if len(recordedHits) > 0:
            recordedHit = recordedHits[0]
            print("Checking dot in frame " + str(videoStream.frameNr))
            testHandleHit(recordedHit, basename, videoStream.frameNr, yamlFilenameList)

    if len(yamlFilenameList) != 0:
        print("Error: Following dots were not detected: " + str(yamlFilenameList))
    lazer.release()

    return videoStream.frameNr


def testHandleHit(recordedHit, filename, frameNr, yamlFilenameList):
    yamlFilename = "tests/" + filename + "_" + str(frameNr) + '_info.yaml'
    yamlFilename2 = "tests/" + filename + "_" + str(frameNr) + '_hit.info.yaml'
    if yamlFilename in yamlFilenameList:
        yamlFilenameList.remove(yamlFilename)
    elif yamlFilename2 in yamlFilenameList:  # FIXME workaround for different filenames
        yamlFilename = yamlFilename2
        yamlFilenameList.remove(yamlFilename)
    else:
        print("Err: Found dot with no testcase at frame " + str(frameNr))
        print("  " + yamlFilename)
        return

    #if not os.path.isfile(yamlFilename):
    #    print("Error: dot detectd, but no tastecase for " + yamlFilename)
    #    return

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


def writeVideoInfo(filename):
    print("Write info of file: " + filename)

    videoStream = FileVideoStream(threaded=True, endless=False)
    videoStream.initFile(filename)
    lazer = Lazer(videoStream, saveFrames=False, saveHits=True)

    while True:
        isTrue = lazer.nextFrame()  # gets next frame, and creates mask
        if not isTrue:
            break
        lazer.detectAndDrawHits()  # all in one for now

    lazer.release()


def old(filename):
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


    lazer.release()

