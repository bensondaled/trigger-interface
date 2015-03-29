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
                ts = (ts,time.time(),time.clock())
            self.current = im
            self.currentts = ts
            self.available = True
    def metadata(self):
        return {}

        
class psVideoCapture(object):
    def __init__(self, idx, resolution, frame_rate):
        self.vc = cv2.VideoCapture(idx)
        self.vc.set(cv.CV_CAP_PROP_FPS, frame_rate)
        self.vc.set(cv.CV_CAP_PROP_FRAME_WIDTH, resolution[0])
        self.vc.set(cv.CV_CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.idx = idx
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
        self.READING = True
        Thread(target=self.continuous_read, args=()).start()
        self.available = False
        time.sleep(0.1)
    def continuous_read(self):
        while self.READING:
            val = False
            while val == False:
                val,fr = self.vc.read()
            self.current = fr
            self.currentts = (time.time(),time.clock())
            self.available = True
    def release(self):
        self.READING = False
        time.sleep(0.02)
        self.vc.release()
    def metadata(self):
        return dict(idx=self.idx,resolution=self.resolution,frame_rate=self.frame_rate)
        
class psVideoCaptureAPI(object):
    def __init__(self, idx, resolution, frame_rate, color_mode, vflip=False, gain=60, exposure=24, wbal_red=50, wbal_blue=50, wbal_green=50):
        self.resolution = resolution
        self.psresolution = {(320,240):PS_SMALL, (640,480):PS_LARGE}[self.resolution]
        self.frame_rate = frame_rate
        self.vflip = vflip
        self.color_mode = color_mode
        self.gain = gain
        self.exposure = exposure
        self.wbal_red = wbal_red
        self.wbal_blue = wbal_blue
        self.wbal_green = wbal_green
        self.pscolor_mode = [mc.CLEYE_GREYSCALE, mc.CLEYE_COLOR][self.color_mode]
        self.vc = mc.Ps3Eye(idx, self.pscolor_mode, self.psresolution, self.frame_rate)
        settings = [ (mc.CLEYE_AUTO_GAIN, 0), \
                 (mc.CLEYE_AUTO_EXPOSURE, 0),\
                 (mc.CLEYE_AUTO_WHITEBALANCE,0),\
                 (mc.CLEYE_GAIN, gain), \
                 (mc.CLEYE_EXPOSURE, exposure),\
                 (mc.CLEYE_WHITEBALANCE_RED,wbal_red),\
                 (mc.CLEYE_WHITEBALANCE_BLUE,wbal_blue),\
                 (mc.CLEYE_WHITEBALANCE_GREEN,wbal_green),\
                 (mc.CLEYE_VFLIP, self.vflip)
                 ]
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
        self.READING = True
        Thread(target=self.continuous_read, args=()).start()
        self.available = False
        time.sleep(0.2)
    def continuous_read(self):
        newdims = [self.resolution[1],self.resolution[0]]
        if self.color_mode == mc.CLEYE_COLOR:
            newdims.append(4)
        while self.READING:
            val = False
            while not val:
                val,fr = self.vc.get_frame()
            self.current = np.fromstring(fr, np.dtype('uint8')).reshape(newdims)
            self.currentts = (time.time(),time.clock())
            self.available = True
    def release(self):
        self.READING = False
        time.sleep(0.02)
        del self.vc
    def metadata(self):
        return dict(resolution = self.resolution,\
        frame_rate = self.frame_rate,\
        vflip = self.vflip,\
        color_mode = self.color_mode,\
        gain = self.gain,\
        exposure = self.exposure,\
        wbal_red = self.wbal_red,\
        wbal_blue = self.wbal_blue,\
        wbal_green = self.wbal_green)
        
class Camera(object):
    BW = 0
    COLOR = 1
    PSEYE = 0
    PG = 1
    PSEYE_NEW = 2
    def __init__(self, idx=0, resolution=(320,240), frame_rate=50, color_mode=BW, cam_type=PSEYE, **kwargs):
        self.idx = idx
        self.resolution = resolution
        self.frame_rate = frame_rate
        self.color_mode = color_mode
        self.cam_type = cam_type
        
        if self.cam_type == self.PSEYE:
            self.vc = psVideoCapture(idx, self.resolution, self.frame_rate)
        elif self.cam_type == self.PSEYE_NEW:
            self.vc = psVideoCaptureAPI(idx, self.resolution, self.frame_rate, self.color_mode, **kwargs)
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
        selfdic = dict(idx=self.idx, resolution=self.resolution, frame_rate=self.frame_rate, color_mode=self.color_mode, cam_type=self.cam_type)
        return dict(cam=selfdic, vc=self.vc.metadata())

if __name__ == '__main__':
    c = Camera(idx=0, resolution=PS_SMALL, frame_rate=100, cam_type=Camera.PSEYE_NEW, color_mode=Camera.COLOR)
    lastts = 0
    while True:
        fr,ts = c.read()
        if ts:
            #cv2.imshow('a',fr)
            #x=cv2.waitKey(1)
            print ts-lastts
            lastts = ts
            #if x == ord('q'):
            #    break
