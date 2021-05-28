import unittest
from detector import Detector
import cv2

Basepath = 'test/data/'

class DetectorTest(unittest.TestCase):
    def test_glare(self):
        filename = "test-glare.jpg"
        capture = cv2.VideoCapture(Basepath + filename)
        ok, frame = capture.read()
        self.assertTrue(ok)
        detector = Detector(thresh=14)
        detector.initFrame(frame)

        # needs to find some glare
        glare = detector.findGlare()
        self.assertGreater(len(glare), 1)


    def test_hit(self):
        filename = "test-glare.jpg"
        capture = cv2.VideoCapture(Basepath + filename)
        ok, frame = capture.read()
        self.assertTrue(ok)
        detector = Detector(thresh=14)
        detector.initFrame(frame)

        # need to find a hit
        hits = detector.findHits(minRadius=1.0)
        self.assertTrue(len(hits) == 1)

        # more precise
        self.assertEqual(hits[0].x, 845)
        self.assertEqual(hits[0].y, 753)
        self.assertEqual(hits[0].radius, 157)


    def test_target(self):
        filename = "test-target.jpg"
        capture = cv2.VideoCapture(Basepath + filename)
        ok, frame = capture.read()
        self.assertTrue(ok)
        detector = Detector(thresh=14)
        detector.initFrame(frame)

        # need to find a target
        contours, reliefs = detector.findTargets(targetThresh=60)

        # its actually like 5
        self.assertGreater(len(reliefs), 0)

        # more precise
        self.assertEqual(reliefs[0].centerX, 558)
        self.assertEqual(reliefs[0].centerY, 614)
        self.assertEqual(reliefs[0].w, 290)

