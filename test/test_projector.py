import unittest
from detector import Detector
from projector import Projector
import cv2

Basepath = 'test/data/'

class ArurcoTest(unittest.TestCase):
    def test_aruco_projection(self):
        projector = Projector()
        projector.initFrame()
        projector.drawAruco()
        frame = projector.frame

        detector = Detector(thresh=14)
        detector.initFrame(frame)

        (corners, ids, rejected) = detector.findAruco()
        self.assertEqual(len(corners), 4)
        self.assertEqual(len(ids), 4)
        #print("A: {}".format(len(corners)))
        #print("B: {}".format(len(rejected)))

        #cv2.imshow("A", frame)
        #cv2.waitKey(0)
