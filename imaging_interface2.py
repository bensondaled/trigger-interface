#natives
import json
import os
pjoin = os.path.join
import time as pytime
#numpy, scipy, matplotlib
import numpy as np 
import pylab as pl
import matplotlib.cm as mpl_cm
from matplotlib import path as mpl_path
#opencv
import cv2
cv = cv2.cv
#custom
from core.daq import DAQ, Trigger
from core.cameras import Camera
from threading import Thread

class Experiment(object):
    def __init__(self, name=None, camera1=None, camera2=None, daq=None, trigger=None, data_dir='.', trial_duration=3., stim_delay=1.):
        self.name = name
        self.data_dir = data_dir
        self.make_exp_dir()
        self.trig = trigger
        
        if type(camera1) == Camera:
            self.camera1 = camera1
            self.camera1.read()
        else:
            self.camera1 = None
        
        if type(camera2) == Camera:
            self.camera2 = camera2
            self.camera2.read()
        else:
            self.camera2 = None
        
        if type(daq) == DAQ:
            self.daq = daq
        elif daq==None:
            self.daq = DAQ(mode=DAQ.DIGITAL, port_digital="Dev1/Port0/Line1")
        else:
            raise Exception('No valid DAQ supplied.')

        self.trial_duration = trial_duration
        self.stim_delay = stim_delay
        cv2.namedWindow('Camera')
        
        self.SAVING = False
        self.PAUSED = False
        self.time1,self.time2,self.trigtime = [],[],None
        self.trial_n = 0
        self.frame1 = np.zeros(self.camera1.resolution)
        self.frame2 = np.zeros(self.camera2.resolution)
    def make_exp_dir(self):
        if self.name == None:
            self.name = pytime.strftime("%Y%m%d_%H%M%S")
        if os.path.isdir(pjoin(self.data_dir,self.name)):
            i = 1
            while os.path.isdir(pjoin(self.data_dir,self.name+'_%i'%i)):
                i += 1
            self.name = self.name+'_%i'%i
        os.mkdir(pjoin(self.data_dir,self.name))
        self.save_dir = pjoin(self.data_dir,self.name)
    def next_frame(self):
        frame1, timestamp1 = self.camera1.read()
        if frame1!=None:
            self.frame1,self.timestamp1 = frame1,timestamp1
            self.new1 = True
        frame2, timestamp2 = self.camera2.read()
        if frame2!=None:
            self.frame2,self.timestamp2 = frame2,timestamp2
            self.new2 = True
    def send_trigger(self):
        self.trigtime = pytime.time()
        self.daq.trigger(self.trig)
        self.trigger_sent = True
    def thread_trigger(self):
        while True:
            if pytime.time()-self.save_start >= self.stim_delay and not self.trigger_sent:
                self.send_trigger()
                break
    def thread_save_cam1(self):
        while self.SAVING:
            frame1, timestamp1 = self.camera1.read()
            if frame1!=None:
                self.writer1.write(frame1)
                self.time1.append(timestamp1)
    def thread_save_cam2(self):
        while self.SAVING:
            frame2, timestamp2 = self.camera2.read()
            if frame2!=None:
                self.writer2.write(frame2)
                self.time2.append(timestamp2)
    def save_current(self):
        if self.new1:
            self.writer1.write(self.frame1)
            self.time1.append(self.timestamp1)
            self.new1 = False
        if self.new2:
            self.writer2.write(self.frame2)
            self.time2.append(self.timestamp2)
            self.new2 = False
    def make_writers(self):
        wname1 = pjoin(self.save_dir,self.trial_name + 'cam1.avi')
        wname2 = pjoin(self.save_dir,self.trial_name + 'cam2.avi')
        self.writer1 = cv2.VideoWriter(wname1,0,30,frameSize=self.camera1.resolution,isColor=False)
        self.writer2 = cv2.VideoWriter(wname2,0,30,frameSize=self.camera2.resolution,isColor=self.camera2.color_mode)
    def start_trial(self):
        self.trial_n += 1
        self.trial_name = pjoin(self.name+'_%02d_'%self.trial_n)
        self.make_writers()
        self.time1,self.time2,self.trigtime = [],[],None
        self.save_start = pytime.time()
        self.trigger_sent = False
        self.SAVING = True
        pytime.sleep(0.025)
        Thread(target=self.thread_save_cam1).start()
        Thread(target=self.thread_save_cam2).start()
        Thread(target=self.thread_trigger).start()
    def end_trial(self):
        np.savez(pjoin(self.save_dir,self.trial_name+'timestamps'), time1=self.time1, time2=self.time2, trigger=self.trigtime)
        self.writer1.release()
        self.writer2.release()
        self.SAVING = False
    def query_trial(self):
        elapsed = pytime.time()-self.save_start
        if elapsed > self.trial_duration:
            self.end_trial()
    def step(self):
        if self.SAVING:
            #self.save_current()
            self.query_trial()
        
        elif not self.SAVING:
            self.next_frame()
            c = cv2.waitKey(1)
            if not self.PAUSED:
                cv2.imshow('Camera1',self.frame1)
                cv2.imshow('Camera2',self.frame2)
            if c == ord('p'):
                self.PAUSED = not self.PAUSED
            elif c == ord('t'):
                self.start_trial()
            elif c == ord('q'):
                return False
            
        return True
    def run(self):
        cv2.namedWindow('Camera1')
        cv2.namedWindow('Camera2')
        cv2.moveWindow('Camera1', 5,5)
        cv2.moveWindow('Camera2', 5+self.camera1.resolution[0],5)
        cont = True
        while cont:
            cont = self.step()
        self.end()
        print "Experiment ended."
    def end(self):
        cv2.destroyAllWindows()
        self.camera1.release()
        self.camera2.release()
        self.daq.release()

if __name__=='__main__':
    pass
