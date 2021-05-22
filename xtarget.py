import cv2 as cv
from collections import deque
import argparse

from lazer import Lazer
from model import *
from videotests import *
from gfxutils import *

import curses
from threading import Thread
from playback import Playback
from videostream import FileVideoStream, CamVideoStream

import logging

logger = logging.getLogger(__name__)


def showFrame(filename, frameNr):
    print("Show frame: " + filename + " at " + str(frameNr))
    videoFileConfig = readVideoFileConfig(filename)

    videoStream = FileVideoStream(threaded=False, endless=False)
    videoStream.setCrop(videoFileConfig['crop'])
    videoStream.initFile(filename)

    lazer = Lazer(videoStream, thresh=videoFileConfig['thresh'], saveFrames=False, saveHits=False)

    videoStream.setFrame(frameNr)
    lazer.nextFrame()
    lazer.displayFrame()

    key = cv.waitKey(0)  # any key quits
    lazer.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="Play a video file")
    ap.add_argument("--cam", help="Capture from webcam id (starting at 0)")
    ap.add_argument("--test", help="Perform analysis of test-videos and validate (slow)", action='store_true')
    ap.add_argument("--testQuick", help="Perform analysis of test-pics and validate (fast)", action='store_true')

    ap.add_argument("--write", help="Write hits from video file as jpg+yaml files")
    
    ap.add_argument("--showframe", help="Show a specific frame (--framenr) of a video")
    ap.add_argument("--framenr", help="frame nr for showframe", type=int)

    # options
    ap.add_argument("--saveHits", help='Option: Save jpg+yaml of all detected hits', action='store_true', default=False)
    ap.add_argument("--saveFrames", help='Option: Save jpg+yaml of every frame', action='store_true', default=False)
    ap.add_argument("--curses", help='Show curses ui in terminal for webcam settings (broken)', action='store_true', default=False)

    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.video is not None:
        filename = args.video
        videoFileConfig = readVideoFileConfig(filename)

        videoStream = FileVideoStream(threaded=False, endless=True)
        if not videoStream.initFile(filename):
            return
        videoStream.setCrop(videoFileConfig['crop'])

        playback = Playback(videoStream, thresh=videoFileConfig['thresh'], saveFrames=args.saveFrames, saveHits=args.saveHits)
        playback.init()
        playback.play()

    elif args.cam is not None:
        camId = int(args.cam)
        videoStream = CamVideoStream(threaded=True)
        videoStream.initCam(camId, resolution={'width': 1920, 'height': 1080})

        playback = Playback(videoStream, saveFrames=args.saveFrames, saveHits=args.saveHits)
        playback.init()
        playback.play()

    elif args.test:
        doTests()
    elif args.testQuick:
        doTestsQuick()

    elif args.showframe is not None:
        filename = args.showframe
        showFrame(filename, args.framenr)

    elif args.write:
        filename = args.write
        writeVideoInfo(filename)


def startCursesThread():
    data = CamConfig()
    gui = Gui(data)
    gui.initCurses()
    gui.run()

if __name__ == "__main__":
    main()