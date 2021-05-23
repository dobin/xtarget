# import the necessary packages
from threading import Thread
import sys
import cv2
import time
from queue import Queue

class InputStream():
    def __init__(self, path):
        self.path = path
        if isinstance(path, int):
            self.isCam = True
        else:
            self.isCam = False


    def initStream(self):
        if self.isCam:
            # use CAP_DSHOW for cams, as it has a MUCH faster initialization
            # Alternatively, CAP_MSMF may provide better performance once initialized.
            self.capture = cv2.VideoCapture(self.path, cv2.CAP_DSHOW)
        else:
            self.capture = cv2.VideoCapture(self.path)

    def read(self):
        # return next frame in the queue
        pass


    def start(self):
        pass


    def release(self):
        pass


class SimpleInputStream(InputStream):
    def __init__(self, path):
        super().__init__(path)
    
    def read(self):
        return self.capture.read()

    def release(self):
        self.capture.release()


class QueueInputStream(InputStream):
    def __init__(self, path):
        super().__init__(path)

        self.stopped = False
        self.transform = False

        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=30)
        # intialize thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        # start a thread to read frames from the file video stream
        self.thread.start()
        return self

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                break

            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                (grabbed, frame) = self.capture.read()

                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.stopped = True
                    
                # if there are transforms to be done, might as well
                # do them on producer thread before handing back to
                # consumer thread. ie. Usually the producer is so far
                # ahead of consumer that we have time to spare.
                #
                # Python is not parallel but the transform operations
                # are usually OpenCV native so release the GIL.
                #
                # Really just trying to avoid spinning up additional
                # native threads and overheads of additional
                # producer/consumer queues since this one was generally
                # idle grabbing frames.
                if self.transform:
                    frame = self.transform(frame)

                # add the frame to the queue
                self.Q.put((grabbed, frame))
            else:
                time.sleep(0.1)  # Rest for 10ms, we have a full queue

        self.capture.release()

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    # Insufficient to have consumer use while(more()) which does
    # not take into account if the producer has reached end of
    # file stream.
    def running(self):
        return self.more() or not self.stopped

    def more(self):
        # return True if there are still frames in the queue. If stream is not stopped, try to wait a moment
        tries = 0
        while self.Q.qsize() == 0 and not self.stopped and tries < 5:
            time.sleep(0.1)
            tries += 1

        return self.Q.qsize() > 0

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        # wait until stream resources are released (producer thread might be still grabbing frame)
        self.thread.join()

    def release(self):
        self.stop()
        self.capture.release()
