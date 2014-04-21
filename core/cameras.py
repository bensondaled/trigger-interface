import numpy as np
import os
import cv2
cv = cv2.cv
import time
import pylab as pl
import json
import sys

class Camera(object):
    BW = 0
    COLOR = 1
    def __init__(self, idx=0, resolution=(320,240), frame_rate=50, color_mode=BW):
        self.resolution = resolution
        self.frame_rate = frame_rate
        self.color_mode = color_mode

        try:
            self.vc = cv2.VideoCapture(idx)
        except:
            raise Exception('Video capture from camera failed to initialize.')
            sys.exit(1)

        self.vc.set(cv.CV_CAP_PROP_FPS, self.frame_rate)
        self.vc.set(cv.CV_CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.vc.set(cv.CV_CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        time.sleep(0.1)
        self.vc.read()
    def read(self):
        success,frame = self.vc.read()
        if self.color_mode==self.BW:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return (frame.astype(np.uint8), time.time())
    def release(self):
        self.vc.release()
    def metadata(self):
        md = {}
        md['resolution'] = self.resolution
        md['frame_rate'] = self.frame_rate
        md['color_mode'] = self.color_mode

        return md
