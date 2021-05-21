import curses
import time
import sys

from model import CamConfig


class CursesUi(object):
    def __init__(self):
        self.camConfig = CamConfig()
        self.selectedLine = 3
        self.stdscr = None

    def initCurses(self):
        self.stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        self.stdscr.keypad(True)
        self.stdscr.nodelay(1)
        self.stdscr.clear()

    def endCurses(self):
        curses.nocbreak()
        self.stdscr.keypad(False)
        curses.echo()
        curses.endwin()


    def run(self):
        self.stdscr.clear()
        self.stdscr.addstr(0, 0, 'xTarget')
        self.stdscr.addstr(3, 3, 'Threshold: {}'.format(self.camConfig.thresh))
        self.stdscr.addstr(4, 3, 'Exposure : {}'.format(self.camConfig.exposure))
        self.stdscr.addstr(5, 3, 'Gain     : {}'.format(self.camConfig.gain))
        self.stdscr.addstr(6, 3, 'AutoExpo : {}'.format(self.camConfig.autoExposure))

        self.stdscr.addstr(self.selectedLine, 0, '>')

        self.stdscr.refresh()
        c = self.stdscr.getch()
        if c == curses.KEY_UP:
            if self.selectedLine > 3:
                self.selectedLine -= 1
        elif c == curses.KEY_DOWN:
            if self.selectedLine < 6:
                self.selectedLine += 1

        value = 0
        if c == curses.KEY_LEFT:
            value = -1
        elif c == curses.KEY_RIGHT:
            value = 1
        if value != 0:
            if self.selectedLine == 3:
                self.camConfig.thresh += value
            if self.selectedLine == 4:
                self.camConfig.exposure += value
            if self.selectedLine == 5:
                self.camConfig.gain += (value * 50)
            if self.selectedLine == 6:
                if value == 1:
                    self.camConfig.autoExposure = 0.25
                else: 
                    self.camConfig.autoExposure = -1.0

        if value != 0:
            return self.camConfig
        else:
            return None

#def main(stdscr):
#    camConfig = camConfig()
#    gui = Gui(camConfig)
#    gui.run(stdscr)
#
#curses.wrapper(main)

def main2():
    camConfig = CamConfig()
    gui = Gui(camConfig)
    gui.initCurses()
    while True:
        gui.run()


#main2()