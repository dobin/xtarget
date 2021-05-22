import unittest
import time

from fps import Fps

class FpsTest(unittest.TestCase):
    def test_fps(self):
        fps = Fps()
        fps.ticklistSize = 3

        fps.tack()
        time.sleep(0.1)
        fps.tack()
        time.sleep(0.1)
        fps.tack()
        time.sleep(0.1)
        fps.tack()
        time.sleep(0.1)

        fps = fps.get()

        # around 10
        self.assertTrue(fps > 8 and fps < 11)
