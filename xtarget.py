import cv2 as cv
from collections import deque
from skimage.metrics import structural_similarity as ssim
import argparse

from lazer import Lazer
from model import *
from tests import *
from gfxutils import *

from ui import Gui
import curses
from threading import Thread
from playback import Playback
from videostream import FileVideoStream, CamVideoStream


def showFrame(filename, frameNr):
    print("Analyzing file: " + filename)
    lazer = Lazer(showVid=True, showGlare=True)
    lazer.initFile(filename)

    cv.namedWindow('Video') 
    lazer.setFrame(frameNr)
    hasFrame = lazer.nextFrame()

    lazer.detectAndDrawHits(staticImage=True)
    lazer.displayFrame()

    key = cv.waitKey(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="Play a video file <file>")
    ap.add_argument("-c", "--cam", help="Capture from webcam id <id>")

    ap.add_argument("-t", "--test", help="Perform analysis of test videos and validate (slow)", action='store_true')
    ap.add_argument("-q", "--testQuick", help="Perform analysis of test pics and validate (fast)", action='store_true')

    #ap.add_argument("-w", "--write", help="write", action='store_true')
    #ap.add_argument("-s", "--showframe", help="showframe", action='store_true')
    #ap.add_argument("-n", "--nr", help="frame nr", type=int)
    #ap.add_argument("--camid", help="Cam", type=int)

    # options
    ap.add_argument("--saveHits", action='store_true', default=False)
    ap.add_argument("--saveFrames", action='store_true', default=False)
    ap.add_argument("--curses", action='store_true', default=False)

    args = ap.parse_args()

    if args.video is not None:
        filename = args.video
        videoFileConfig = readVideoFileConfig(filename)

        videoStream = FileVideoStream(threaded=False, endless=True)
        videoStream.initFile(filename)
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

    elif args.write:
        writeVideoInfo(filename)
    elif args.showframe:
        showFrame(filename, args.nr)


def startCursesThread():
    data = CamConfig()
    gui = Gui(data)
    gui.initCurses()
    gui.run()

if __name__ == "__main__":
    main()