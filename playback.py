import cv2 as cv
import numpy as np

from lazer import Lazer
from cursesui import CursesUi
from model import Mode
from gfxutils import *


class Playback(object):
    """Opens a window to play back a video/cam via VideoStream and uses Lazer for detection and visualization"""

    def __init__(self, videoStream, thresh=14, saveFrames=False, saveHits=False, cursesEnabled=False):
        """Call init() before use"""
        self.videoStream = videoStream
        self.cursesEnabled = cursesEnabled
        self.lazer = Lazer(videoStream, thresh=thresh, saveFrames=saveFrames, saveHits=saveHits, mode=Mode.intro)
        
        self.cursesUi = None
        self.isPaused = False
        self.cropModeEnabled = False
        self.targetModeEnabled = False

        self.initClick()


    def initClick(self):
        """Reset all UI selection related data"""
        self.trackerLocA = None
        self.trackerLocB = None
        self.trackX = 0
        self.trackY = 0
        self.hitRadius = 0


    def init(self):
        """Init I/O"""
        if self.cursesEnabled:
            self.cursesUi = CursesUi()
            self.cursesUi.initCurses()
            
        cv.namedWindow('Video')
        cv.setMouseCallback("Video", self.clickTrack)


    def play(self):
        """Play the video/cam source from self.videoStream with hit detection as a window in a endless loop"""
        while True:
            self.lazer.nextFrame()  # gets next frame, and creates mask
            self.drawTrackings()  # local trackings. needs to be before displayFrame
            self.lazer.displayFrame()  # draw ui n stuff

            # input
            self.handleCurses()  # curses is optional
            if self.isPaused:
                key = cv.waitKey(0)  # wait forever as pause
            else:
                key = cv.waitKey(2)
            self.handleKey(key)
            if key == ord('q'):  # quit
                break

        # end
        if self.cursesEnabled:
            self.cursesUi.endCurses()
        self.lazer.release()
        cv.destroyAllWindows()
        print("Quitting nicely...")


    def handleKey(self, key):
        """Handle input keys by invoking their appropriate actions"""
        if key == ord('c'):  # Crop selection mode
            self.cropModeEnabled = not self.cropModeEnabled
            if not self.cropModeEnabled and self.trackerLocB != None:
                # exited crop mode, set the resulting crop
                crop = [ 
                        self.trackerLocA,
                        self.trackerLocB,
                ]
                self.lazer.videoStream.setCrop(crop)

        if key == ord('t'):  # Target selection mode
            self.targetModeEnabled = not self.targetModeEnabled
            if not self.targetModeEnabled and self.trackerLocB != None:
                self.lazer.setTargetCenter(self.trackerLocA[0], self.trackerLocA[1], self.hitRadius)

        if key == ord(' '):  # Mode
            self.lazer.resetDynamic()
            if self.lazer.mode == Mode.intro:
                self.lazer.changeMode(Mode.main)
            elif self.lazer.mode == Mode.main:
                self.lazer.changeMode(Mode.intro)

        # only applicable for video files
        if key == ord('d'): # back
            self.lazer.videoStream.setFrame(self.lazer.videoStream.frameNr-1)
            self.lazer.resetDynamic()
        elif key == ord('e'): # back 10
            self.lazer.videoStream.setFrame(self.lazer.videoStream.frameNr-11)
            self.lazer.resetDynamic()
        elif key == ord('f'): # forward
            #lazer.nextFrame()
            pass
        elif key == ord('p'):  # pause
            self.isPaused = not self.isPaused
            self.lazer.resetDynamic()

        # Note: when we press a key in paused mode, we actually go to the next 
        # frame. We have to manually go one back every time with setFrame(lazer.FrameNr)
        if key == ord('s'):  # save frame
            if self.isPaused:
                self.lazer.setFrame(self.lazer.videoStream.frameNr)
            self.lazer.saveCurrentFrame(epilog=".live")
        if key == ord('j'):  # decrease threshhold
            if self.isPaused:
                self.lazer.setFrame(self.lazer.videoStream.frameNr)
            self.lazer.thresh -= 1
        if key == ord('k'):  # increase threshhold
            if self.isPaused:
                self.lazer.setFrame(self.lazer.videoStream.frameNr)
            self.lazer.thresh += 1


    def handleCurses(self):
        if not self.cursesEnabled:
            return
            
        if self.lazer.frameNr % 5 == 0:  # rate limit curses i/o for now
            camConfig = self.cursesUi.run()
            if camConfig != None:
                self.lazer.updateCamSettings(camConfig)


    def clickTrack(self, event, x, y, flags, param):
        """Mouse callback used to handle mouse events based on mode"""
        if self.cropModeEnabled or self.targetModeEnabled:
            if event == cv.EVENT_LBUTTONUP:
                self.trackerLocB = (x,y)
                self.hitRadius = calculateDistance(self.trackerLocA[0], self.trackerLocA[1], x, y)
                logger.info("Click Up  : " + str(self.trackerLocB) + " Radius: " + str(self.hitRadius))
            if event == cv.EVENT_LBUTTONDOWN:
                self.initClick()
                self.trackerLocA = (x,y)
                self.trackX = x
                self.trackY = y
                logger.info("Click Down: " + str(self.trackerLocA))
            elif event == cv.EVENT_MOUSEMOVE:
                self.trackX = x
                self.trackY = y

    def drawTrackings(self):
        """Draw temporary UI selection onto the frame for clickTrack"""
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
