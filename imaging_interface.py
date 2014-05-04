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
    def __init__(self, camera=None, daq=None, mask_names=('WHEEL','EYE'),  motion_mask='WHEEL', movement_query_frames=10, movement_std_thresh=1.5, trigger_cycle=None, inter_trial_min=10.0, n_trials=-1, resample=1):
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
            self.camera.read()
        else:
            raise Exception('No valid camera supplied.')
        
        if type(daq) == DAQ:
            self.daq = daq
        elif daq==None:
            self.daq = DAQ(mode=DAQ.DIGITAL)
        else:
            raise Exception('No valid DAQ supplied.')
        self.analog_daq = DAQ(mode=DAQ.ANALOG)

        # Set static parameters
        self.trigger_cycle = trigger_cycle
        self.mask_names = mask_names
        self.resample = resample

        # Set variable parameters
        self.param_names = ['movement_std_threshold', 'movement_query_frames', 'inter_trial_min', 'wheel_translation','wheel_stretch','eye_translation','eye_stretch']
        self.params = {}
        self.params['movement_std_threshold'] = movement_std_thresh
        self.params['movement_query_frames'] = movement_query_frames
        self.params['inter_trial_min'] = inter_trial_min
        self.params['wheel_translation'] = 50
        self.params['wheel_stretch'] = 2
        self.params['eye_translation'] = 50
        self.params['eye_stretch'] = 1
        

        # Setup interface
        pl.ion()
        self.fig = pl.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_ylim([-1,100])
        self.plotdata = {m:self.ax.plot(np.arange(self.params['movement_query_frames']),np.zeros(self.params['movement_query_frames']), c)[0] for m,c in zip(self.mask_names,['r-','b-'])}  
        self.plotline, = self.ax.plot(np.arange(self.params['movement_query_frames']), [self.params['movement_std_threshold']+self.params['wheel_translation'] for _ in range(self.params['movement_query_frames'])], 'k--')
        self.window = 'Interface'
        self.status = 'Status'
        self.controls = 'Controls'
        self.params_win = 'Parameters'
        cv2.namedWindow(self.window, cv.CV_WINDOW_NORMAL)
        cv2.namedWindow(self.status, cv.CV_WINDOW_NORMAL)
        cv2.namedWindow(self.controls, cv.CV_WINDOW_NORMAL)
        cv2.namedWindow(self.params_win, cv.CV_WINDOW_NORMAL)
        cv2.moveWindow(self.window, 210, 0)
        cv2.moveWindow(self.status, 960, 0)
        cv2.moveWindow(self.controls, 0, 0)
        cv2.moveWindow(self.params_win, 1020, 0)
        self.disp_controls()
        for pn in self.param_names:
            cv2.createTrackbar(pn,self.params_win,int(self.params[pn]), 100, self.update_trackbar_params)
        self.update_trackbar_params(self)
        
        # Set initial variables
        self.masks = {}
        self.mask_idxs = {}
        self.motion_mask = motion_mask

        # Run interactive init
        self.init(trials=n_trials)
    def update_trackbar_params(self, _):
        for param in self.param_names:
            self.params[param] = cv2.getTrackbarPos(param,self.params_win)
        self.params['wheel_translation'] -= 50
        self.params['eye_translation'] -= 50
        self.plotline.set_ydata([self.params['movement_std_threshold']+self.params['wheel_translation'] for i in range(self.params['movement_query_frames'])])
    def disp_controls(self):
        img = np.ones((150,170))*255
        lab_origin = 10
        val_origin = 120
        textsize = 0.4
        textheight = 25
    
        items = {}
        items['Pause'] = 'p'
        items['Continue (go)'] = 'g'
        items['Redo trial'] = 'r'
        items['Quit'] = 'q'
        items['Manual Trigger'] = 't'

        for idx,item in enumerate(items):
            cv2.putText(img,item+':', (lab_origin,textheight+idx*textheight), cv2.FONT_HERSHEY_SIMPLEX, textsize, (0,0,0)) 
            cv2.putText(img,items[item], (val_origin,textheight+idx*textheight), cv2.FONT_HERSHEY_SIMPLEX, textsize, (0,0,0)) 
    
        cv2.imshow(self.controls, img)
    def update_status(self):
        self.status_img = np.ones((400,300))*255
        lab_origin = 10
        val_origin = 120
        textsize = 0.4
        textheight = 30
    
        items = {}
        items['Since last'] = str(round(pytime.time()-self.last_trial_off, 3))
        items['Trials done'] = str(self.trial_count)
        items['Paused'] = str(self.TRIAL_PAUSE)
        items['Last trigger'] = str(self.trigger_cycle.current.name)

        for idx,item in enumerate(items):
            cv2.putText(self.status_img,item+':', (lab_origin,textheight+idx*textheight), cv2.FONT_HERSHEY_SIMPLEX, textsize, (0,0,0)) 
            cv2.putText(self.status_img,items[item], (val_origin,textheight+idx*textheight), cv2.FONT_HERSHEY_SIMPLEX, textsize, (0,0,0)) 
    
        cv2.imshow(self.status, self.status_img)
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
        self.monitor_img_set = {n:None for n in self.mask_names}
        self.monitor_vals = {m:[] for m in self.mask_names}
        self.TRIAL_ON = False
        self.TRIAL_PAUSE = False
        self.last_trial_off = pytime.time()
        self.frame_count = 0
        
        # save metadata
        dic = {}
        dic['name'] = self.name
        dic['initial_params'] = self.params
        dic['camera'] = self.camera.metadata()
        dic['daq'] = self.daq.metadata()
        dic['trigger_cycle'] = self.trigger_cycle.metadata()
        dic['resample'] = self.resample
        f = open(os.path.join(self.name,"metadata.json"), 'w')
        f.write("%s"%json.dumps(dic))
        f.close()
        
        self.update_status()
        # run some initial frames 
        for i in range(self.params['movement_query_frames']):
            self.next_frame()
    def set_masks(self):
        for m in self.mask_names:
            print "Please select mask: %s."%m
            frame, timestamp = self.camera.read()
            pl.figure()
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
    def end(self):
        try:
            self.camera.release()
            self.daq.release()
            self.analog_daq.release()
            cv2.destroyAllWindows()
            pl.close(self.fig)
        except:
            pass
    def query_for_trigger(self):
        if pytime.time()-self.last_trial_off < self.params['inter_trial_min']:
            return False
        return self.monitor_vals[self.motion_mask][-1] < self.params['movement_std_threshold']
    def monitor_frame(self, frame):
        for mask in self.mask_names:
            if self.monitor_img_set[mask] != None:
                self.monitor_img_set[mask] = np.dstack([self.monitor_img_set[mask], frame])
                
                # only hold on to as many as you'll look at:
                if np.shape(self.monitor_img_set[mask])[-1]>self.params['movement_query_frames']:
                    self.monitor_img_set[mask] = self.monitor_img_set[mask][...,-self.params['movement_query_frames']:]
                
                mask_idxs = self.mask_idxs[mask]
                if mask==self.motion_mask:
                    std_pts = np.std(self.monitor_img_set[mask][mask_idxs[0],mask_idxs[1],:], axis=1)
                    self.monitor_vals[mask].append(np.mean(std_pts))
                else:
                    mean_pts = np.mean(self.monitor_img_set[mask][mask_idxs[0],mask_idxs[1],:], axis=1)
                    self.monitor_vals[mask].append(np.mean(mean_pts))

                if len(self.monitor_vals[mask])>self.params['movement_query_frames']:
                    self.monitor_vals[mask] = self.monitor_vals[mask][-self.params['movement_query_frames']:]
            else:
                self.monitor_img_set[mask] = frame
        if len(self.monitor_vals['EYE']) > 0:
            tr = Trigger(msg=self.normalize(self.monitor_vals['EYE'][-1], minn=self.analog_daq.minn, maxx=self.analog_daq.maxx))
            self.analog_daq.trigger(tr)
        
        toshow_w = np.array(self.monitor_vals['WHEEL']) * self.params['wheel_stretch']/10. + self.params['wheel_translation']
        toshow_e = np.array(self.monitor_vals['EYE'])* self.params['eye_stretch']/10. + self.params['eye_translation']
        if len(toshow_w) != self.params['movement_query_frames']:
            toshow_w = np.append(toshow_w, np.array([None for _ in range(self.params['movement_query_frames']-len(toshow_w))]))
            toshow_e = np.append(toshow_e, np.array([None for _ in range(self.params['movement_query_frames']-len(toshow_e))]))
        self.plotdata['WHEEL'].set_ydata(toshow_w)
        self.plotdata['EYE'].set_ydata(toshow_e)
        self.fig.canvas.draw()
    def normalize(self, val, minn, maxx):
        return val/1000. * (maxx-minn) + minn
    def next_frame(self):
        frame, timestamp = self.camera.read()
        self.frame_count += 1
        if self.TRIAL_ON:
            self.writer.write(frame)
            self.time.append(timestamp)
        if not self.frame_count % self.resample:
            if not self.TRIAL_ON and not self.TRIAL_PAUSE:
                self.monitor_frame(frame)
            cv2.imshow(self.window, frame)
    def send_trigger(self):
        self.daq.trigger(self.trigger_cycle.next)
        print "Sent trigger #%i"%(self.trial_count+1)
    def start_trial(self):
        self.TRIAL_ON = pytime.time()
        self.trial_count += 1
        
        self.filename = os.path.join(self.name,'trial%i'%(self.trial_count))
        if os.path.isfile(self.filename):
            i = 1
            while os.path.isfile(os.path.join(self.name,'trial%i_redo%i'%(self.trial_count,i))):
                i += 1
            self.filename = os.path.join(self.name,'trial%i_redo%i.npz'%(self.trial_count,i))
        
        self.writer = cv2.VideoWriter(self.filename+'.avi',0,self.camera.frame_rate,frameSize=self.camera.resolution,isColor=False)
        self.time = []
        self.monitor_img_set = {n:None for n in self.mask_names}
        self.monitor_vals = {m:[] for m in self.mask_names}
    def end_trial(self):
        self.TRIAL_ON = False
        self.last_trial_off = pytime.time()
        np.savez_compressed(self.filename+'.npz', time=self.time)         
        self.writer.release()
        self.filename = None
    def step(self):
        self.next_frame()
        c = cv2.waitKey(1)
        
        if self.TRIAL_ON:
            if pytime.time()-self.TRIAL_ON >= self.trigger_cycle.current.duration:
                self.end_trial()

        if not self.TRIAL_ON: 
            
            if c == ord('p'):
                self.TRIAL_PAUSE = True
                self.update_status()
                
            if c == ord('g'):
                self.TRIAL_PAUSE = False

            if c == ord('r'):
                self.trigger_cycle.redo()
                self.trial_count -= 1

            if c == ord('q') or (self.trial_count==self.trials_total):
                return False
            
            if not self.TRIAL_PAUSE:
                if self.query_for_trigger() or c==ord('t'):
                    self.send_trigger()
                    self.start_trial()
                self.update_status()
        
        return True
    def run(self):
        cont = True
        while cont:
            cont = self.step()
        self.end()
        print "Experiment ended."
        
class TriggerCycle(object):
    def __init__(self, triggers=[]):
        self.triggers = np.array(triggers)
        self.current = Trigger(msg=[0,0,0,0], duration=0.0, name='(no trigger yet)')
    @property
    def next(self):
        n = self.triggers[0]
        self.current = n
        self.triggers = np.roll(self.triggers, -1)
        return n
    def redo(self):
        self.triggers = np.roll(self.triggers, 1)
        self.current = self.triggers[-1]
    def metadata(self):
        md = {}
        md['triggers'] = [t.metadata() for t in self.triggers]
        return md

if __name__=='__main__':
    cam =  Camera(idx=0, resolution=(320,240), frame_rate=100, color_mode=Camera.BW)

    CS = Trigger(msg=[0,0,1,1], duration=5.0, name='CS')
    US = Trigger(msg=[0,0,0,1], duration=5.0, name='US')
    trigger_cycle = TriggerCycle(triggers=[CS, US, CS, CS])
    
    exp = Experiment(camera=cam, trigger_cycle=trigger_cycle, n_trials=20, resample=5, movement_std_thresh=10)
    exp.run() #'q' can always be used to end the run early. don't kill the process
# display average eyelid after triggers
"""
        ##
        self.writer = cv2.VideoWriter('test1.avi',-1,self.camera.frame_rate,frameSize=self.camera.resolution,isColor=False)
        ts = []
        self.TRIAL_ON = True
        for i in range(400):
            x= self.next_frame()
            ts.append(pytime.time())
        print 1/np.mean(np.array(ts)[1:] - np.array(ts)[:-1])
        self.writer.release()
        return False
        ##
"""
