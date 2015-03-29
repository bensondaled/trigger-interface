#natives
import json
import os
pjoin = os.path.join
import sys
import time as pytime
from threading import Thread
import subprocess as sbp
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

class Experiment(object):
    def __init__(self, name=None, camera1=None, camera2=None, camera3=None, daq=None, trigger=None, data_dir='.', trial_duration=3., stim_delay=1.):
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
            
        if type(camera3) == Camera:
            self.camera3 = camera3
            self.camera3.read()
        else:
            self.camera3= None            
        
        if type(daq) == DAQ:
            self.daq = daq
        elif daq==None:
            self.daq = DAQ(mode=DAQ.DIGITAL, port_digital="Dev1/Port0/Line1")
        else:
            raise Exception('No valid DAQ supplied.')

        self.trial_duration = trial_duration
        self.stim_delay = stim_delay
        
        self.SAVING = False
        self.PAUSED = False
        if self.camera1:
            self.frame1 = np.zeros(self.camera1.resolution)
        if self.camera2:
            self.frame2 = np.zeros(self.camera2.resolution)
        if self.camera3:
            self.frame3 = np.zeros(self.camera3.resolution)
        self.save_metadata()
    def make_exp_dir(self):
        if self.name == None:
            self.name = pytime.strftime("%Y%m%d_%H%M%S")
        if os.path.isdir(pjoin(self.data_dir,self.name)):
            i = 1
            while os.path.isdir(pjoin(self.data_dir,self.name+'_%i'%i)):
                i += 1
            self.name = self.name+'_%i'%i
        print "Auto-confirmed name: %s"%self.name
        os.mkdir(pjoin(self.data_dir,self.name))
        self.save_dir = pjoin(self.data_dir,self.name)
        self.log_file = open(pjoin(self.save_dir,self.name+'.log'),'a')
        self.trial_n = 0
    def save_metadata(self):
        metadata = dict(name=self.name, data_dir=self.data_dir, trial_duration = self.trial_duration, stim_delay=self.stim_delay, save_dir=self.save_dir)
        metadata['trigger_md'] = self.trig.metadata()
        if self.camera1:
            metadata['camera1_md'] = self.camera1.metadata()
        if self.camera2:
            metadata['camera2_md'] = self.camera2.metadata()
        if self.camera3:
            metadata['camera3_md'] = self.camera3.metadata()
        with open(pjoin(self.save_dir,self.name+'_metadata.json'),'w') as f:
            f.write('%s'%json.dumps(metadata))
    def change_name(self):
        oldname = self.name
        newname = ''
        while newname == '':
            newname = raw_input('Enter new name: ').lower()
        self.log('name changed from %s to %s'%(oldname,newname))
        self.log_file.close()
        self.name = newname
        self.make_exp_dir()
        self.save_metadata()
        print "Name changed."
        self.log('name changed from %s to %s (adjusted to %s)'%(oldname,newname,self.name))
    def log(self, msg):
        print >> self.log_file, '%0.9f %s' %(pytime.time(),msg)
        self.log_file.flush()
    def next_frame(self):
        if self.camera1:
            frame1, timestamp1 = self.camera1.read()
            if frame1!=None:
                self.frame1,self.timestamp1 = frame1,timestamp1
                self.new1 = True
        if self.camera2:
            frame2, timestamp2 = self.camera2.read()
            if frame2!=None:
                self.frame2,self.timestamp2 = frame2,timestamp2
                self.new2 = True
        if self.camera3:
            frame3, timestamp3 = self.camera3.read()
            if frame3!=None:
                self.frame3,self.timestamp3 = frame3,timestamp3
                self.new3 = True
    def resize(self,frame,width=860):
        fsize = frame.shape
        if len(fsize)>2:
            fsize = fsize[:len(fsize)-1]
        rsf = frame.shape[1]/float(width)
        newshape = np.round(np.array(fsize)[::-1]/rsf).astype(int)
        frame = cv2.resize(frame, tuple(newshape))
        return frame
    def send_trigger(self):
        self.trigtime = [pytime.time(),pytime.clock()]
        self.daq.trigger(self.trig)
        self.trigger_sent = True
        self.log('sent trigger')
    def thread_trigger(self):
        while True:
            if pytime.time()-self.save_start >= self.stim_delay and not self.trigger_sent:
                self.send_trigger()
                break
    def thread_save_cam1(self):
        if self.camera1 == None:
            return
        while self.SAVING:
            frame1, timestamp1 = self.camera1.read()
            if frame1!=None:
                self.writer1.write(frame1)
                self.time1.append(timestamp1)
        self.writer1.release()
    def thread_save_cam2(self):
        if self.camera2 == None:
            return
        while self.SAVING:
            frame2, timestamp2 = self.camera2.read()
            if frame2!=None:
                self.writer2.write(frame2)
                self.time2.append(timestamp2)
        self.writer2.release()
    def thread_save_cam3(self):
        if self.camera3 == None:
            return
        while self.SAVING:
            frame3, timestamp3 = self.camera3.read()
            if frame3!=None:
                self.writer3.write(frame3)
                self.time3.append(timestamp3)
        self.writer3.release()
    def make_writers(self):
        wname1 = pjoin(self.save_dir,self.trial_name + 'cam1.avi')
        wname2 = pjoin(self.save_dir,self.trial_name + 'cam2.avi')
        wname3 = pjoin(self.save_dir,self.trial_name + 'cam3.avi')
        if self.camera1 != None:
            self.writer1 = cv2.VideoWriter(wname1,0,30,frameSize=self.camera1.resolution,isColor=False)
        if self.camera2 != None:
            self.writer2 = cv2.VideoWriter(wname2,0,30,frameSize=self.camera2.resolution,isColor=self.camera2.color_mode)
        if self.camera3 != None:
            self.writer3 = cv2.VideoWriter(wname3,0,30,frameSize=self.camera3.resolution,isColor=self.camera3.color_mode)
    def start_trial(self):
        self.trial_n += 1
        self.trial_name = pjoin(self.name+'_%02d_'%self.trial_n)
        self.make_writers()
        self.time1,self.time2,self.time3,self.trigtime = [],[],[],None
        self.save_start = pytime.time()
        self.trigger_sent = False
        self.SAVING = True
        pytime.sleep(0.025)
        Thread(target=self.thread_save_cam1).start()
        Thread(target=self.thread_save_cam2).start()
        Thread(target=self.thread_save_cam3).start()
        Thread(target=self.thread_trigger).start()
        self.log('trial %i started'%self.trial_n)
    def end_trial(self):
        np.savez(pjoin(self.save_dir,self.trial_name+'timestamps'), time1=self.time1, time2=self.time2, time3=self.time3, trigger=self.trigtime)
        self.SAVING = False
        self.log('trial %i ended'%self.trial_n)
        pytime.sleep(0.030)
    def query_trial(self):
        elapsed = pytime.time()-self.save_start
        if elapsed > self.trial_duration:
            self.end_trial()
    def step(self):
        if self.SAVING:
            self.query_trial()
        
        elif not self.SAVING:
            self.next_frame()
            c = cv2.waitKey(1)
            if not self.PAUSED:
                cv2.imshow('Camera1',self.resize(self.frame1, width=860))
                cv2.imshow('Camera2',self.resize(self.frame2, width=320))
                cv2.imshow('Camera3',self.resize(self.frame3, width=320))
            if c == ord('p'):
                self.PAUSED = not self.PAUSED
            elif c == ord('t'):
                self.start_trial()
            elif c == ord('e'):
                self.change_name()
            elif c == ord('r'):
                cv2.destroyAllWindows()
                self.log('replaying last trial')
                out = sbp.call(['python','mapping_playback.py',self.name,'play'])
                self.log('done replay')
                self.place_windows()
            elif c == ord('a'):
                cv2.destroyAllWindows()
                self.log('plotting last trial')
                out = sbp.call(['python','mapping_playback.py',self.name,'plot'])
                self.log('done plot')
                self.place_windows()
            elif c in [ord(str(i)) for i in xrange(1,10)]:
                n = str(range(1,10)[[ord(str(i)) for i in xrange(1,10)].index(c)])
                cv2.destroyAllWindows()
                self.log('plotting last %s trials'%n)
                out = sbp.call(['python','mapping_playback.py',self.name,'plot',n])
                self.log('done plot')
                self.place_windows()
            elif c == ord('q'):
                return False
            
        return True
    def place_windows(self):
        if self.camera1:
            cv2.namedWindow('Camera1')
            cv2.moveWindow('Camera1', 5,5)
        if self.camera2:
            cv2.namedWindow('Camera2')
            cv2.moveWindow('Camera2', 600,5)
        if self.camera3:
            cv2.namedWindow('Camera3')
            cv2.moveWindow('Camera3', 600,300)
    def run(self):
        self.log('started run')
        self.place_windows()
        cont = True
        while cont:
            cont = self.step()
        self.end()
        print "Experiment ended."
    def end(self):
        self.log('quit')
        cv2.destroyAllWindows()
        if self.camera1:
            self.camera1.release()
        if self.camera2:
            self.camera2.release()
        if self.camera3:
            self.camera3.release()
        self.daq.release()
        self.log_file.close()

if __name__=='__main__':
    pass
