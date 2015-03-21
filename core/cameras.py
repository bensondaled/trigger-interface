import numpy as np
import os
import cv2
cv = cv2.cv
import time
import pylab as pl
import json
import sys
import multicam as mc
from threading import Thread
try:
    import flycapture2 as fc2
except:
    fc2 = None

PS_SMALL = mc.CLEYE_QVGA
PS_LARGE = mc.CLEYE_VGA    

class fcVideoCapture(object):
    def __init__(self):
        self.c = fc2.Context()
        self.c.connect(*self.c.get_camera_from_index(0))
        self.c.set_timestamping(True)
        self.c.start_capture()
        self.img = fc2.Image()
        self.c.retrieve_buffer(self.img)
        self.first_ts = self.img.get_timestamp()
        self.on = True
        self.start()
    def read(self):
        if self.available:
            self.available = False
            return self.current,self.currentts
        else:
            return None,None
    def diff(self, ts1, ts2):
        return (ts2['cycle_secs']-ts1['cycle_secs']) + (ts2['cycle_count']-ts1['cycle_count'])/8000.
    def release(self):
        self.on = False
        time.sleep(0.1)
        self.c.stop_capture()
        self.c.disconnect()
    def start(self):
        Thread(target=self.continuous_read, args=()).start()
        self.available = False
        time.sleep(0.1)
    def continuous_read(self):
        while True:
            if not self.on:
                break
            im = None
            while not np.any(im):
                self.c.retrieve_buffer(self.img)
                im = np.array(self.img)
                ts = self.img.get_timestamp()
                ts = self.diff(self.first_ts,ts)
                ts = (ts,time.time())
            self.current = im
            self.currentts = ts
            self.available = True

        
class psVideoCapture(object):
    def __init__(self, idx, resolution, frame_rate):
        self.vc = cv2.VideoCapture(idx)
        self.vc.set(cv.CV_CAP_PROP_FPS, frame_rate)
        self.vc.set(cv.CV_CAP_PROP_FRAME_WIDTH, resolution[0])
        self.vc.set(cv.CV_CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.resolution = resolution
        self.frame_rate = frame_rate
        
        self.start()
    def read(self):
        if self.available:
            self.available = False
            return self.current,self.currentts
        else:
            return None,None
    def start(self):
        Thread(target=self.continuous_read, args=()).start()
        self.available = False
        time.sleep(0.1)
    def continuous_read(self):
        while True:
            val = False
            while val == False:
                val,fr = self.vc.read()
            self.current = fr
            self.currentts = time.time()
            self.available = True
    def release(self):
        self.vc.release()
        
class psVideoCaptureAPI(object):
    def __init__(self, idx, resolution, frame_rate):
        self.resolution = resolution
        self.dims = {PS_SMALL:[320,240],PS_LARGE:[640,480]}[self.resolution]
        self.frame_rate = frame_rate
        self.vc = mc.Ps3Eye(0, mc.CLEYE_MONO_PROCESSED, self.resolution, self.frame_rate)
        settings = [ (mc.CLEYE_AUTO_GAIN, 1), \
                 (mc.CLEYE_AUTO_EXPOSURE, 1),\
                 (mc.CLEYE_AUTO_WHITEBALANCE, 1)]
        self.vc.configure(settings)
        self.vc.start()
        
        self.start()
    def read(self):
        if self.available:
            self.available = False
            return self.current,self.currentts
        else:
            return None,None
    def start(self):
        Thread(target=self.continuous_read, args=()).start()
        self.available = False
        time.sleep(0.1)
    def continuous_read(self):
        while True:
            val = False
            while val == False:
                val,fr = self.vc.get_frame()
            self.current = np.fromstring(fr, np.dtype('uint8')).reshape(self.dims[::-1])
            self.currentts = time.time()
            self.available = True
    def release(self):
        self.vc.release()
        
class Camera(object):
    BW = 0
    COLOR = 1
    PSEYE = 0
    PG = 1
    PSEYE_NEW = 2
    def __init__(self, idx=0, resolution=(320,240), frame_rate=50, color_mode=BW, cam_type=PSEYE):
        self.resolution = resolution
        self.frame_rate = frame_rate
        self.color_mode = color_mode
        self.cam_type = cam_type
        
        if self.cam_type == self.PSEYE:
            self.vc = psVideoCapture(idx, self.resolution, self.frame_rate)
        elif self.cam_type == self.PSEYE_NEW:
            self.vc = psVideoCaptureAPI(idx, self.resolution, self.frame_rate)
        elif self.cam_type == self.PG:
            self.vc = fcVideoCapture()
            
    def read(self):
        if self.vc.available:
            frame,timestamp = self.vc.read()
            if self.color_mode==self.BW and self.cam_type==self.PSEYE:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return (frame.astype(np.uint8), timestamp)
        else:
            return None,None

    def release(self):
        self.vc.release()
    def metadata(self):
        return self.__dict__
