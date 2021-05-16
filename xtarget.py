import cv2 as cv
from collections import deque
import numpy as np
from skimage.metrics import structural_similarity as ssim
import argparse

from lazer import Lazer
from model import *
from tests import *
from gfxutils import *

class Playback(object):
    def __init__(self):
        # for extract_coordinates_callback
        self.lazer = None
        
        self.isPaused = False
        self.cropModeEnabled = False
        self.targetModeEnabled = False

        self.initClick()


    def initClick(self):
        self.trackerLocA = None
        self.trackerLocB = None
        self.trackX = 0
        self.trackY = 0
        self.hitRadius = 0


    def click_track(self, event, x, y, flags, param):
        if self.cropModeEnabled or self.targetModeEnabled:
            if event == cv.EVENT_LBUTTONUP:
                self.trackerLocB = (x,y)
                self.hitRadius = calculateDistance(self.trackerLocA[0], self.trackerLocA[1], x, y)
                print("Up  : " + str(self.trackerLocB) + " Radius: " + str(self.hitRadius))
            if event == cv.EVENT_LBUTTONDOWN:
                self.initClick()
                self.trackerLocA = (x,y)
                self.trackX = x
                self.trackY = y
                print("Down: " + str(self.trackerLocA))
            elif event == cv.EVENT_MOUSEMOVE:
                self.trackX = x
                self.trackY = y


    def drawTrackings(self):
        if self.cropModeEnabled and self.trackerLocA != None:
            if self.trackerLocB == None:
                cv.rectangle(self.lazer.frame, 
                    self.trackerLocA, 
                    (self.trackX, self.trackY), 
                    (0,255,0), 2)
            else:
                cv.rectangle(self.lazer.frame, 
                    self.trackerLocA, 
                    self.trackerLocB, 
                    (0,255,0), 2)

        if self.targetModeEnabled and self.trackerLocA != None:
            if self.trackerLocB == None:
                center = np.array([self.trackerLocA[0], self.trackerLocA[1]], dtype=np.int64)
                pt_on_circle = np.array([self.trackX, self.trackY], dtype=np.int64)
                radius = int(np.linalg.norm(pt_on_circle-center))
                cv.circle(self.lazer.frame, (center[0], center[1]), radius, (0,255,0), 2)
            else:
                center = np.array([self.trackerLocA[0], self.trackerLocA[1]], dtype=np.int64)
                pt_on_circle = np.array([self.trackerLocB[0], self.trackerLocB[1]], dtype=np.int64)
                radius = int(np.linalg.norm(pt_on_circle-center))
                cv.circle(self.lazer.frame, (center[0], center[1]), radius, (0,255,0), 2)


    def play(self, filename, saveFrames=False, saveHits=False, camId=None):
        print("Analyzing file: " + str(filename))
        lazer = Lazer(showVid=True, saveFrames=saveFrames, saveHits=saveHits, endless=True)
        self.lazer = lazer

        if camId != None:
            lazer.initCam(camId)
        else: 
            lazer.initFile(filename)

        cv.namedWindow('Video')
        cv.setMouseCallback("Video", self.click_track)

        while True:
            lazer.nextFrame()  # gets next frame, and creates mask

            lazer.detectAndDrawHits()

            self.drawTrackings()  # needs to be before displayFrame
            lazer.displayFrame()

            # input
            if self.isPaused:
                key = cv.waitKey(0)  # wait forever
            else:
                key = cv.waitKey(2)

            if key == ord('c'):  # Crop image
                self.cropModeEnabled = not self.cropModeEnabled

                if not self.cropModeEnabled and self.trackerLocB != None:
                    crop = [ 
                            self.trackerLocA,
                            self.trackerLocB,
                    ]
                    self.lazer.setCrop(crop)

            if key == ord('t'):  # Target Mode
                self.targetModeEnabled = not self.targetModeEnabled

                if not self.targetModeEnabled and self.trackerLocB != None:
                    self.lazer.setCenter(self.trackerLocA[0], self.trackerLocA[1], self.hitRadius)

            if key == ord('m'):  # Mode
                if lazer.mode == Mode.intro:
                    lazer.changeMode(Mode.main)
                elif lazer.mode == Mode.main:
                    lazer.changeMode(Mode.intro)

            elif key == ord('d'): # back
                lazer.setFrame(lazer.frameNr-1)
                lazer.init()
            elif key == ord('e'): # back 10
                lazer.setFrame(lazer.frameNr-11)
                lazer.init()
            elif key == ord('f'): # forward
                #lazer.nextFrame()
                pass
            elif key == ord(' '):  # pause
                self.isPaused = not self.isPaused
                lazer.init()

            if key == ord('s'):  # save frame
                if self.isPaused:
                    lazer.setFrame(lazer.frameNr)
                lazer.saveCurrentFrame(epilog=".live")
            if key == ord('j'):  # decrease threshhold
                if self.isPaused:
                    lazer.setFrame(lazer.frameNr)
                lazer.thresh -= 1
            if key == ord('k'):  # increase threshhold
                if self.isPaused:
                    lazer.setFrame(lazer.frameNr)
                lazer.thresh += 1

            if key == ord('q'):
                break

        lazer.release()
        print("")


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

    args = ap.parse_args()

    filename = args.file
    if args.video:
        playback = Playback()
        playback.play(filename, saveFrames=args.saveFrames, saveHits=args.saveHits)
    elif args.test:
        doTests()
    elif args.testQuick:
        doTestsQuick()
    elif args.write:
        writeVideoInfo(filename)
    elif args.showframe:
        showFrame(filename, args.nr)
    elif args.cam:
        playback = Playback()
        playback.play('cam', saveFrames=args.saveFrames, saveHits=args.saveHits, camId=args.camid)


if __name__ == "__main__":
    main()