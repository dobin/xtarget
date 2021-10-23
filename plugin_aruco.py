import logging
import cv2
import json

logger = logging.getLogger(__name__)


class PluginAruco(object):
    def __init__(self):
        # every frame
        self.arucoCorners = []
        self.arucoIds = []

        # zero to four identified arucos
        self.arucoCornersAll = {}
        self.arucoIdsAll = {}


    def init(self):
        self.arucoCorners = []
        self.arucoIds = []
        self.arucoCornersAll = {}
        self.arucoIdsAll = {}


    def handle(self, frame, corners, ids, rejected):
        self.arucoCorners = corners
        self.arucoIds = ids

        # if we found all four, nothing more to do
        if len(self.arucoCornersAll) == 4:
            return

        # no corners, nothing to do
        if len(corners) == 0:
            return corners, ids

        for idx, corner in enumerate(self.arucoCorners):
            id = ids[idx][0]
            self.arucoCornersAll[id] = corner
            self.arucoIdsAll[id] = ids[idx]

        if len(self.arucoCornersAll) == 4:
            logger.info("Found all four aruco corners")


    def draw(self, frame):
        # draw all found in current frame
        for corner in self.arucoCorners:
            a = (int(corner[0][0][0]), int(corner[0][0][1]))
            b = (int(corner[0][2][0]), int(corner[0][2][1]))
            cv2.rectangle(
                frame,
                a,
                b,
                (0, 255, 0), 2)

        # draw all identified
        for corner in list(self.arucoCornersAll.values()):
            a = (int(corner[0][0][0]), int(corner[0][0][1]))
            b = (int(corner[0][2][0]), int(corner[0][2][1]))
            cv2.rectangle(
                frame,
                a,
                b,
                (255, 255, 0), 2)


# disabled as we should have an order
#        # draw rectangle for all four found
#        if len(self.arucoCorners) == 4:
#            # draw aruco area
#            a = (int(self.arucoCorners[3][0][0][0]), int(self.arucoCorners[3][0][0][1]))
#            b = (int(self.arucoCorners[0][0][2][0]), int(self.arucoCorners[0][0][2][1]))
#            cv2.rectangle(
#                frame,
#                a,
#                b,
#                (0, 255, 255), 2)
