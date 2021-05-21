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
    ap.add_argument("-v", "--video", help="video", action='store_true')
    ap.add_argument("-t", "--test", help="test", action='store_true')
    ap.add_argument("-w", "--write", help="write", action='store_true')
    ap.add_argument("-s", "--showframe", help="showframe", action='store_true')
    ap.add_argument("-q", "--testQuick", action='store_true')
    ap.add_argument("-c", "--cam", action='store_true')

    ap.add_argument("-f", "--file", help="file", type=str)
    ap.add_argument("-n", "--nr", help="frame nr", type=int)
    ap.add_argument("--camid", help="Cam", type=int)

    ap.add_argument("--saveHits", action='store_true', default=False)
    ap.add_argument("--saveFrames", action='store_true', default=False)

    ap.add_argument("--curses", action='store_true', default=False)

    args = ap.parse_args()

    filename = args.file
    if args.video:
        videoFileConfig = readVideoFileConfig(filename)

        videoStream = FileVideoStream(threaded=False, endless=True)
        videoStream.initFile(filename)
        videoStream.setCrop(videoFileConfig['crop'])

        playback = Playback(videoStream, thresh=videoFileConfig['thresh'], saveFrames=args.saveFrames, saveHits=args.saveHits)
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
    elif args.cam:
        camId = args.camid

        videoStream = CamVideoStream(threaded=False)
        videoStream.initCam(camId)

        playback = Playback(videoStream, saveFrames=args.saveFrames, saveHits=args.saveHits)
        playback.init()
        playback.play()


def readVideoFileConfig(filename):
    config = {
        'crop': None,
        'thresh': 14,
    }

    # check for crop settings for file
    vidYaml = filename +'.yaml'
    if os.path.isfile(vidYaml):
        print("Opening video config file...")
        with open(vidYaml) as file:
            vidYamlData = yaml.load(file, Loader=yaml.FullLoader)

            crop = []
            if 'x1' in vidYamlData:
                crop.append((vidYamlData['x1'], vidYamlData['y1']))
                crop.append((vidYamlData['x2'], vidYamlData['y2']))
            config['crop'] = crop

            if 'thresh' in vidYamlData:
                config['thresh'] = vidYamlData['thresh']

            return config

    return config


def startCursesThread():
    data = CamConfig()
    gui = Gui(data)
    gui.initCurses()
    gui.run()

if __name__ == "__main__":
    main()