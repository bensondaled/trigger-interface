#natives
import json
import os
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

class Experiment(object):
    def __init__(self, camera=None, daq=None, mask_names=('WHEEL','EYE'), movement_query_frames=10, movement_std_thresh=1.5, trigger_cycle=None, inter_trial_min=10.0, n_trials=-1, resample=1, monitor_vals_display=100):
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
        self.monitor_vals_display = monitor_vals_display

        # Set variable parameters
        self.param_names = ['movement_std_threshold', 'movement_query_frames', 'inter_trial_min', 'wheel_translation','wheel_stretch','eye_translation','eye_stretch']
        self.params = {}
        self.params['movement_std_threshold'] = movement_std_thresh
        self.params['movement_query_frames'] = movement_query_frames
        self.params['inter_trial_min'] = inter_trial_min
        self.params['wheel_translation'] = 50
        self.params['wheel_stretch'] = 25
        self.params['eye_translation'] = 50
        self.params['eye_stretch'] = 25
        

        # Setup interface
        pl.ion()
        self.fig = pl.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_ylim([-1, 255])
        self.plotdata = {m:self.ax.plot(np.arange(self.monitor_vals_display),np.zeros(self.monitor_vals_display), c)[0] for m,c in zip(self.mask_names,['r-','b-'])} 
        self.plotline, = self.ax.plot(np.arange(self.monitor_vals_display), np.repeat(self.params['movement_std_threshold'], self.monitor_vals_display), 'r--')
        self.window = 'Interface'
        self.status = 'Status'
        self.controls = 'Controls'
        self.params_win = 'Parameters'
        cv2.namedWindow(self.window, cv.CV_WINDOW_NORMAL)
        cv2.namedWindow(self.status, cv.CV_WINDOW_NORMAL)
        cv2.namedWindow(self.controls, cv.CV_WINDOW_NORMAL)
        cv2.namedWindow(self.params_win, cv.CV_WINDOW_NORMAL)
        cv2.moveWindow(self.window, 0, 0)
        cv2.moveWindow(self.status, 0, 600)
        cv2.moveWindow(self.controls, 800, 0)
        cv2.moveWindow(self.params_win, 800, 600)
        self.disp_controls()
        for pn in self.param_names:
            cv2.createTrackbar(pn,self.params_win,int(self.params[pn]), 100, self.update_trackbar_params)
        self.update_trackbar_params(self)
        
        # Set initial variables
        self.masks = {}
        self.mask_idxs = {}

        # Run interactive init
        self.init(trials=n_trials)
    def update_trackbar_params(self, _):
        for param in self.param_names:
            self.params[param] = cv2.getTrackbarPos(param,self.params_win)
        self.params['wheel_translation'] -= 50
        self.params['eye_translation'] -= 50
        self.params['wheel_stretch'] /= 25.
        self.params['eye_stretch'] /= 25.
        self.plotline.set_ydata(np.repeat(self.params['movement_std_threshold'], self.monitor_vals_display))
    def disp_controls(self):    
        items = {}
        items['Pause'] = 'p'
        items['Continue (go)'] = 'g'
        items['Redo trial'] = 'r'
        items['Quit'] = 'q'
        items['Manual Trigger'] = 't'
        
        lab_origin = 10
        val_origin = 120
        textsize = 0.4
        textheight = 25
        img = np.ones((round(textheight*len(items)*1.1),170))*255

        for idx,item in enumerate(items):
            cv2.putText(img,item+':', (lab_origin,textheight+idx*textheight), cv2.FONT_HERSHEY_SIMPLEX, textsize, (0,0,0)) 
            cv2.putText(img,items[item], (val_origin,textheight+idx*textheight), cv2.FONT_HERSHEY_SIMPLEX, textsize, (0,0,0)) 
    
        cv2.imshow(self.controls, img)
    def update_status(self):
        items = {}
        items['Since last'] = round(pytime.time()-self.last_trial_off, 3)
        items['Trials done'] = self.trial_count
        items['Paused'] = self.TRIAL_PAUSE
        items['Last trigger'] = self.trigger_cycle.current.name
        if len(self.monitor_vals['EYE']):
            items['Eyelid Value'] = self.monitor_vals['EYE'][-1]
        else:
            items['Eyelid Value'] = '(none yet)'
        items['Frame Rate'] = self.inst_frame_rate

        for item in items:
            items[item] = str(items[item])
      
        lab_origin = 10
        val_origin = 120
        textsize = 0.4
        textheight = 25
        self.status_img = np.ones((round(textheight*len(items)*1.1),300))*255
        
        for idx,item in enumerate(items):
            cv2.putText(self.status_img,item+':', (lab_origin,textheight+idx*textheight), cv2.FONT_HERSHEY_SIMPLEX, textsize, (0,0,0)) 
            cv2.putText(self.status_img,items[item], (val_origin,textheight+idx*textheight), cv2.FONT_HERSHEY_SIMPLEX, textsize, (0,0,0)) 
    
        cv2.imshow(self.status, self.status_img)
    def init(self, trials):
        self.name = pytime.strftime("%Y%m%d_%H%M%S")
        os.mkdir(self.name)

        # set up frame rate details
        self.last_timestamp = pytime.time()
        self.inst_frame_rate = None

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
    def update_framerate(self, timestamp):
        fr = 1/(timestamp - self.last_timestamp)
        self.inst_frame_rate = fr
        self.last_timestamp = timestamp
    def set_masks(self):
        for m in self.mask_names:
            frame, timestamp = self.camera.read()
            pl.figure()
            pl.title("Select mask: %s."%m)
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
        return self.monitor_vals['WHEEL'][-1] < self.params['movement_std_threshold']
    def monitor_frame(self, frame):
        for mask in self.mask_names:
            if self.monitor_img_set[mask] != None:
                self.monitor_img_set[mask] = np.dstack([self.monitor_img_set[mask], frame])
                
                # only hold on to as many as you'll use for computation:
                if np.shape(self.monitor_img_set[mask])[-1]>self.params['movement_query_frames']:
                    self.monitor_img_set[mask] = self.monitor_img_set[mask][...,-self.params['movement_query_frames']:]
                
                mask_idxs = self.mask_idxs[mask]
                if mask=='WHEEL':
                    std_pts = np.std(self.monitor_img_set[mask][mask_idxs[0],mask_idxs[1],:], axis=1)
                    wval = np.mean(std_pts) * self.params['wheel_stretch'] + self.params['wheel_translation']
                    self.monitor_vals[mask].append(wval)
                elif mask=='EYE':
                    mean_pts = np.mean(self.monitor_img_set[mask][mask_idxs[0],mask_idxs[1],-1])
                    eyval = np.mean(mean_pts) * self.params['eye_stretch'] + self.params['eye_translation']
                    self.monitor_vals[mask].append(eyval)

                # only hold on to as many as you'll display:
                if len(self.monitor_vals[mask])>self.monitor_vals_display:
                    self.monitor_vals[mask] = self.monitor_vals[mask][-self.monitor_vals_display:]
            else:
                self.monitor_img_set[mask] = frame
        
        self.update_analog_daq()
        self.update_plots()
    def update_analog_daq(self):
        if len(self.monitor_vals['EYE']) > 0:
            val = self.monitor_vals['EYE'][-1]
            val = self.normalize(val, oldmin=0., oldmax=255. * self.params['eye_stretch'] + self.params['eye_translation'], newmin=self.analog_daq.minn, newmax=self.analog_daq.maxx)          
            tr = Trigger(msg=val)
            self.analog_daq.trigger(tr)
    def normalize(self, val, oldmin, oldmax, newmin, newmax):
        return ((val-oldmin)/oldmax) * (newmax-newmin) + newmin
    def update_plots(self):
        toshow_w = np.array(self.monitor_vals['WHEEL'])
        toshow_e = np.array(self.monitor_vals['EYE'])
        if len(toshow_w) != self.monitor_vals_display:
            toshow_w = np.append(toshow_w, np.repeat(None, self.monitor_vals_display-len(toshow_w)))
            toshow_e = np.append(toshow_e, np.repeat(None, self.monitor_vals_display-len(toshow_e)))
        self.plotdata['WHEEL'].set_ydata(toshow_w)
        self.plotdata['EYE'].set_ydata(toshow_e)
        self.fig.canvas.draw()
    def next_frame(self):
        frame, timestamp = self.camera.read()
        self.update_framerate(timestamp)
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
    pass
