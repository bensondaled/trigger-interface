#natives
import json
import os
import time as pytime

#numpy, scipy, matplotlib
import numpy as np 
import pylab as pl
import matplotlib.cm as mpl_cm
from matplotlib import path as mpl_path
import matplotlib.animation as ani

#opencv
import cv2
cv = cv2.cv

#custom
from core.daq import DAQ, Trigger
from core.cameras import Camera

class Experiment(object):
    def __init__(self, camera=None, daq=None, mask_names=('WHEEL','EYE'),  motion_mask='WHEEL', movement_query_frames=20, movement_std_thresh=1.5, trigger=None, inter_trial_min=5.0, n_trials=-1):
        """
        Parameters:
                cameras: [list of] Camera object[s]
                daq: one DAQ object (data acquisition interface)
                save_mode (const int): either NP (numpy) or CV (openCV)
                mask_names (list of str): names for the masks to be selected from monitor_cam
                monitor_cam_idx: index of camera used to monitor with masks
                motion_mask: name of mask used to detect motion
                movement_query_frames: number of past frames to analyze for motion
                movement_std_thresh: standard deviation threshold for movement detection
                trigger: Trigger object for the experiment
        """
        if type(camera) == Camera:
            self.camera = camera
        else:
            raise Exception('No valid camera supplied.')
        
        if type(daq) == DAQ:
            self.daq = daq
        else:
            raise Exception('No valid DAQ supplied.')

        # Set static parameters
        self.trigger = trigger
        self.mask_names = mask_names

        # Set variable parameters
        self.param_names = ['movement_std_threshold', 'movement_query_frames', 'inter_trial_min']
        self.params = {}
        self.params['movement_std_threshold'] = movement_std_thresh
        self.params['movement_query_frames'] = movement_query_frames
        self.params['inter_trial_min'] = inter_trial_min

        # Setup interface
        self.window = 'Interface'
        cv2.namedWindow(self.window, cv.CV_WINDOW_NORMAL)
        for pn in self.param_names:
            cv2.createTrackbar(pn,self.window,int(self.params[pn]), 100, self.update_trackbar_params)

        # Set initial variables
        self.masks = {}
        self.mask_idxs = {}
        self.motion_mask = motion_mask

        # Run interactive init
        self.init(trials=n_trials)
    def update_trackbar_params(self, _):
        for param in self.param_names:
            self.params[param] = cv2.getTrackbarPos(param,self.window)
        print self.params['movement_std_threshold']
    def init(self, trials):
        self.name = pytime.strftime("%Y%m%d_%H%M%S")
        os.mkdir(self.name)

        # set up trial count
        self.trials_total = trials
        if trials == -1:
            self.trials_total = 10**3
        self.trial_count = 0
        
        # ask user for masks and set them
        if len(self.masks)==0:     
            self.set_masks()
        
        # setup containers for acquired data
        self.img_set = None
        self.monitor_img_set = None
        self.time = []
        self.TRIAL_ON = False
        self.TRIAL_PAUSE = False
        self.last_trial_off = pytime.time()
        
        # save metadata
        dic = {}
        dic['name'] = self.name
        dic['initial_params'] = self.params
        dic['camera'] = self.camera.metadata()
        dic['daq'] = self.daq.metadata()
        dic['trigger'] = self.trigger.metadata()
        f = open(os.path.join(self.name,"metadata.json"), 'w')
        f.write("%s"%json.dumps(dic))
        f.close()
        
        # run some initial frames 
        for i in range(self.params['movement_query_frames']):
            self.next_frame()
    def set_masks(self):
        for m in self.mask_names:
            print "Please select mask: %s."%m
            frame, timestamp = self.camera.read()
            pl.imshow(frame, cmap=mpl_cm.Greys_r)
            pts = pl.ginput(0)
            pl.close()
            path = mpl_path.Path(pts)
            mask = np.ones(np.shape(frame), dtype=bool)
            for ridx,row in enumerate(mask):
                for cidx,pt in enumerate(row):
                    if path.contains_point([cidx, ridx]):
                        mask[ridx,cidx] = False
            self.masks[m] = mask
            self.mask_idxs[m] = np.where(mask==False)
    def save(self, cam_idx=None, frame=None):
        np.savez_compressed(os.path.join(self.name,'trial%i'%(self.trial_count)), time=self.time, data=self.img_set.transpose(2,0,1))
        self.img_set = None
        self.time = []
    def end(self):
        try:
            self.camera.release()
            self.daq.release()
            cv2.destroyAllWindows()
        except:
            pass
    def query_for_trigger(self):
        if pytime.time()-self.last_trial_off < self.params['inter_trial_min']:
            return False
        mask_idxs = self.mask_idxs[self.motion_mask]
        std_pts = np.std(self.monitor_img_set[mask_idxs[0],mask_idxs[1],:], axis=0)
        return np.mean(std_pts) < self.params['movement_std_threshold']
    def store_frame(self, frame, timestamp):
        if self.img_set != None:
            self.img_set = np.dstack([self.img_set, frame])
        else:
            self.img_set = frame
        self.time.append(timestamp)
    def monitor_frame(self, frame):
        if self.monitor_img_set != None:
            self.monitor_img_set = np.dstack([self.monitor_img_set, frame])
            
            # only hold on to as many as you'll look at:
            if np.shape(self.monitor_img_set)[-1]>self.params['movement_query_frames']:
                self.monitor_img_set = self.monitor_img_set[...,-self.params['movement_query_frames']:]
        else:
            self.monitor_img_set = frame
    def next_frame(self):
        if not self.TRIAL_PAUSE:
            frame, timestamp = self.camera.read()
            if self.TRIAL_ON:
                self.store_frame(frame, timestamp)
            elif not self.TRIAL_ON:
                self.monitor_frame(frame)
                cv2.imshow(self.window, frame)
    def send_trigger(self):
        self.daq.trigger(self.trigger)
        print "Sent trigger #%i"%(self.trial_count)
    def step(self):
        self.next_frame()
        
        if self.TRIAL_ON:
            if pytime.time()-self.TRIAL_ON >= self.trigger.duration:
                self.TRIAL_ON = False
                self.last_trial_off = pytime.time()
                self.save()

        if not self.TRIAL_ON:           
            c = cv2.waitKey(1)
            
            if c == ord('p'):
                self.TRIAL_PAUSE = True
                
            if c == ord('g'):
                self.TRIAL_PAUSE = False

            if c == ord('q') or (self.trial_count==self.trials_total):
                return False
            
            if not self.TRIAL_PAUSE:
                if self.query_for_trigger():
                    self.send_trigger()
                    self.TRIAL_ON = pytime.time()
                    self.trial_count += 1
                    self.monitor_img_set = None
        
        return True
    def run(self):
        cont = True
        while cont:
            cont = self.step()
        self.end()
        
        
if __name__=='__main__':
    cam =  Camera(idx=0, resolution=(320,240), frame_rate=20, color_mode=Camera.BW)
    trigger = Trigger(msg=[0,0,1,1], duration=5.0)
    daq = DAQ()
    
    exp = Experiment(camera=cam, daq=DAQ(), trigger=trigger)
    exp.run() #'q' can always be used to end the run early. don't kill the process
