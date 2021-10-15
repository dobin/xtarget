import logging
import cv2

logger = logging.getLogger(__name__)


class PluginAruco(object):
    def __init__(self):
        self.arucoCorners = None
        self.arucoIds = None


    def init(self):
        self.arucoCorners = None
        self.arucoIds = None


    def handle(self, frame, corners, ids, rejected):
        if self.arucoCorners is not None:
            return

        for corner in corners:
            a = (int(corner[0][0][0]), int(corner[0][0][1]))
            b = (int(corner[0][2][0]), int(corner[0][2][1]))
            cv2.rectangle(
                frame,
                a,
                b,
                (0, 255, 255), 2)

        if len(corners) != 4:
            return None, None

        self.arucoCorners = corners
        self.arucoIds = ids
        logger.info("Found 4 aruco {} {}".format(len(corners), len(ids)))
        return corners, ids


    def draw(self, frame):
        if self.arucoCorners is None:
            return

        # draw aruco area
        a = (int(self.arucoCorners[3][0][0][0]), int(self.arucoCorners[3][0][0][1]))
        b = (int(self.arucoCorners[0][0][2][0]), int(self.arucoCorners[0][0][2][1]))
        cv2.rectangle(
            frame,
            a,
            b,
            (0, 255, 255), 2)
