import numpy as np 
import time
from cameras import Camera
import os
import json
import cv2
cv = cv2.cv


class Monitor(object):
    def __init__(self, cameras=None, show=True, save_on=True):
        if type(cameras) == Camera:
            self.cameras = [cameras]
        elif type(cameras) == list:
            self.cameras = cameras
        elif cameras == None:
            self.cameras = []
        
        self.show = show
        self.save_on = save_on
        self.windows = self.make_windows()
        
        if self.save_on:
            self.times = [[] for i in self.cameras] 

            self.run_name = time.strftime("%Y%m%d_%H%M%S")
            os.mkdir(self.run_name)
            os.chdir(self.run_name)
            
            self.writers = [cv2.VideoWriter(self.run_name+"-cam%i.avi"%i,0,\
            cam.frame_rate,\
            frameSize=cam.resolution,\
            isColor=False) \
            for i,cam in enumerate(self.cameras)]
            
            dic = {}
            dic['run_name'] = self.run_name
            dic['experiment'] = self.metadata()
            dic['cameras'] = [cam.metadata() for cam in self.cameras]
            
            f = open("%s-metadata.json"%self.run_name, 'w')
            f.write("%s"%json.dumps(dic))
            f.close()
        
    def metadata(self):
        md = {}
        md['n_cameras'] = len(self.cameras)
        return md
    def save(self, cam_idx=None, frame=None):
        self.writers[cam_idx].write(frame)
    def make_windows(self):
        windows = [str(i) for i in range(len(self.cameras))]
        for w in windows:
            cv2.namedWindow(w, cv.CV_WINDOW_NORMAL)
        return windows
    def end(self):
        for cam in self.cameras:
            cam.release()
        for win in self.windows:
            cv2.destroyWindow(win)
            
        if self.save_on:
            [writer.release() for writer in self.writers]
            
            f = open("%s-timestamps.json"%self.run_name, 'w')
            f.write("%s"%json.dumps(self.times))
            f.close()
            self.convert_to_np()
            os.chdir('..')
    def convert_to_np(self):
        for idx,cam in enumerate( [i for i in os.listdir('.') if 'avi' in i] ):
            mov = cv2.VideoCapture(cam)
            valid,frame = mov.read()
            frames = np.zeros((len(self.times[idx]),np.shape(frame)[0],np.shape(frame)[1]),dtype=np.uint8)
            i = 0
            while valid:
                frame = cv2.cvtColor(frame, cv2.cv.CV_RGB2GRAY)
                frames[i,:,:] = frame.astype(np.uint8)
                valid,frame = mov.read()
                i += 1
            np.savez_compressed(cam[:cam.index('.avi')], data=frames.astype(np.uint8), time=self.times[idx])
            mov.release()
    def next_frame(self):
        for cam_idx,win,cam in zip(range(len(self.cameras)),self.windows,self.cameras):
            frame, timestamp = cam.read()
            frame, timestamp = cam.read()
            if self.show:
                cv2.imshow(win, frame)
                c = cv2.waitKey(1)
            if self.save_on:
                self.times[cam_idx].append(timestamp)
                self.save(cam_idx, frame)
        return c
    
    def run(self, duration=999999999.):
        t = time.time()
        c = None
        while time.time()-t < duration and c!=ord('q'):
            c = self.next_frame()
        self.end()

if __name__ == '__main__':
    m = Monitor(cameras=Camera(), show=True, save=False)
    m.run()
