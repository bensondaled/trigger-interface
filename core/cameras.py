import numpy as np
import os
import cv2
cv = cv2.cv
import time
import pylab as pl
import json
import sys
try:
    import flycapture2 as fc2
except:
    fc2 = None

class fcVideoCapture(object):
    def __init__(self):
        self.c = fc2.Context()
        self.c.connect(*c.get_camera_from_index(0))
        self.c.set_timestamping(True)
        self.c.start_capture()
        self.img = fc2.Image()
        c.retrieve_buffer(self.img)
        self.first_ts = self.img.get_timestamp()
    def read(self):
        c.retrieve_buffer(self.img)
        im = np.array(self.img)
        ts = self.img.get_timestamp()
        ts = self.diff(self.first_ts,ts)
        return (im, ts)
    def diff(self, ts1, ts2):
        return (ts2['cycle_secs']-ts1['cycle_secs']) + (ts2['cycle_count']-ts1['cycle_count'])/8000.
    def release(self):
        self.c.stop_capture()
        self.c.disconnect()

class Camera(object):
    BW = 0
    COLOR = 1
    PSEYE = 0
    PG = 1
    def __init__(self, idx=0, resolution=(320,240), frame_rate=50, color_mode=BW, cam_type=None):
        self.resolution = resolution
        self.frame_rate = frame_rate
        self.color_mode = color_mode
        self.cam_type = cam_type
        if cam_type == None:
            self.cam_type = self.PSEYE
        
        if self.cam_type == self.PSEYE:
            try:
                self.vc = cv2.VideoCapture(idx)
            except:
                raise Exception('Video capture from camera failed to initialize.')

            self.vc.set(cv.CV_CAP_PROP_FPS, self.frame_rate)
            self.vc.set(cv.CV_CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.vc.set(cv.CV_CAP_PROP_FRAME_HEIGHT, self.resolution[1])

            time.sleep(0.1)
            self.vc.read()
        elif self.cam_type == self.PG:
            self.vc = fcVideoCapture()
    def read(self):
        if self.cam_type == self.PSEYE:
            success,frame = self.vc.read()
            timestamp = time.time()
        elif self.cam_type == self.PG:
            frame, timestamp = self.vc.read()
            timestamp = (time.time(), timestamp)
        if self.color_mode==self.BW:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return (frame.astype(np.uint8), timestamp)
    def release(self):
        self.vc.release()
    def metadata(self):
        return self.__dict__
