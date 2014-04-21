import numpy as np 
import PySide.QtGui as qt
import PySide.QtCore as qtc
import os
import cv2
cv = cv2.cv
import time as pytime
import pylab as pl
pl.ioff()
import json
import matplotlib.cm as mpl_cm
from matplotlib import path as mpl_path
import matplotlib.animation as ani
from daq import DAQ, Trigger
from cameras import Camera, BW, COLOR
import matplotlib
matplotlib.rcParams['backend.qt4'] = "PySide"
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

class Experiment(object):
    def __init__(self, camera=None, daq=None, mask_names=('WHEEL','EYE'),  motion_mask='WHEEL', movement_query_frames=20, movement_std_thresh=1.5, trigger=None, inter_trial_min=5.0):
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


        # Set initial variables
        self.mask_names = mask_names
        self.masks = {}
        self.mask_idxs = {}
        self.motion_mask = motion_mask
        #self.window = '0'
        #cv2.namedWindow(self.window, cv.CV_WINDOW_NORMAL)
        
        # Set parameters
        self.movement_std_thresh = movement_std_thresh
        self.movement_query_frames = movement_query_frames
        self.trigger = trigger
        self.inter_trial_min = inter_trial_min
        
        # Setup metadata
        self.run_info = self.empty_run_info()

        self.RUNNING = False
    def metadata(self):
        md = {}
        md['movement_std_thresh'] = self.movement_std_thresh
        md['movement_query_frames'] = self.movement_query_frames
        md['inter_trial_min'] = self.inter_trial_min
        
        return md
    def empty_run_info(self):
        runinf = {}
        runinf['trigger_times'] = []
        return runinf
    def new_run(self, new_masks=False,trials=None):         
        if new_masks or len(self.masks)==0:     
            self.set_masks()
        self.trials_total = trials
        if trials == -1:
            self.trials_total = 10**3
        self.trial_count = 0

        self.img_set = None
        self.monitor_img_set = None
        self.time = []

        self.run_name = pytime.strftime("%Y%m%d_%H%M%S")
        
        self.run_info = self.empty_run_info()
        os.mkdir(self.run_name)
        os.chdir(self.run_name)
        
        dic = {}
        dic['run_name'] = self.run_name
        dic['experiment'] = self.metadata()
        dic['camera'] = self.camera.metadata()
        dic['daq'] = self.daq.metadata()
        dic['trigger'] = self.trigger.metadata()
        
        f = open("%s-metadata.json"%self.run_name, 'w')
        f.write("%s"%json.dumps(dic))
        f.close()
        
        self.TRIAL_ON = False
        self.last_trial_off = pytime.time()
        
        self.RUNNING = True

        for i in range(self.movement_query_frames):
            self.next_frame()
    def end_run(self):
        f = open("%s-data.json"%self.run_name, 'w')
        f.write("%s"%json.dumps(self.run_info))
        f.close()
        os.chdir('..')
        self.RUNNING = False
    def make_mask(self):
        win = qt.QMainWindow()
        grid = qt.QGridLayout()
        frame, timestamp = self.camera.read()
        fig = pl.Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.imshow(frame, cmap=mpl_cm.Greys_r)
        grid.addWidget(canvas,0,0)
        win.setLayout(grid)
        win.show()
        pts = pl.ginput(0)
        path = mpl_path.Path(pts)
        mask = np.ones(np.shape(frame), dtype=bool)
        for ridx,row in enumerate(mask):
            for cidx,pt in enumerate(row):
                if path.contains_point([cidx, ridx]):
                    mask[ridx,cidx] = False
        return mask
    def set_masks(self):
        for m in self.mask_names:
            print "Please select mask: %s."%m
            mask = self.make_mask()
            self.masks[m] = mask
            self.mask_idxs[m] = np.where(mask==False)
    def save(self, cam_idx=None, frame=None):
        np.savez_compressed(self.run_name+'_trial%i'%(self.trial_count), time=self.time, data=self.img_set.astype(np.uint8))
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
        if pytime.time()-self.last_trial_off < self.inter_trial_min:
            return False
        mask_idxs = self.mask_idxs[self.motion_mask]
        std_pts = np.std(self.monitor_img_set[:,mask_idxs[0],mask_idxs[1]], axis=0)
        return np.mean(std_pts) < self.movement_std_thresh
    def record_frame(self, frame):
        if self.img_set != None:
            self.img_set = np.append(self.img_set, [frame], axis=0)
        else:
            self.img_set = np.array([frame],dtype=np.uint8)
    def monitor_frame(self, frame):
        if self.monitor_img_set != None:
            self.monitor_img_set = np.append(self.monitor_img_set, [frame], axis=0)
        else:
            self.monitor_img_set = np.array([frame], dtype=np.uint8)
        if np.shape(self.monitor_img_set)[0]>self.movement_query_frames:
            self.monitor_img_set = self.monitor_img_set[-self.movement_query_frames:]
    def next_frame(self):
        frame, timestamp = self.camera.read()
        if self.TRIAL_ON:
            self.record_frame(frame)
            self.time.append(timestamp)
        elif not self.TRIAL_ON:
            self.monitor_frame(frame)
            #cv2.imshow(self.window, frame)
    def send_trigger(self):
        self.daq.trigger(self.trigger)
        self.run_info['trigger_times'].append(pytime.time())
    def step(self):
        if self.RUNNING:
            self.next_frame()
            
            if self.TRIAL_ON:
                if pytime.time()-self.TRIAL_ON >= self.trigger.duration:
                    self.TRIAL_ON = False
                    self.last_trial_off = pytime.time()
                    self.save()

            if not self.TRIAL_ON:           
                c = cv2.waitKey(1)
                if c == ord('q') or (self.trial_count==self.trials_total):
                    self.end_run()
                    return
                    
                if self.query_for_trigger():
                    self.monitor_img_set = None
                    self.send_trigger()
                    self.TRIAL_ON = pytime.time()
                    self.trial_count += 1
                    print "Sent trigger #%i"%(self.trial_count)
                    
class ExpWindow(qt.QWidget):
    def __init__(self):
        qt.QWidget.__init__(self)
        
        # basics
        self.setWindowTitle('Behavioural Setup Data Acquisition Interface')
        self.setFixedWidth(1000)
        self.setFixedHeight(700)
        grid = qt.QGridLayout()
      
        # constants
        self.RUN = 0
        self.PAUSE = 1
        self.END = 2

        # camera image
        pixmap = qt.QPixmap("/Users/Benson/Desktop/a.png")
        frame = qt.QLabel()
        frame.setPixmap(pixmap)
       
        # labels
        self.label1 = qt.QLabel()
        self.label1.setText('nothing yet.')

        # text parameters
        w_std_thresh = qt.QLineEdit()
        w_std_thresh.setObjectName('std_thresh')
        w_trial_num = qt.QLineEdit()
        w_trial_num.setObjectName('trial_num')
        w_total_trials = qt.QLineEdit()
        w_total_trials.setObjectName('total_trials')
        for w in [w_std_thresh, w_trial_num]:
            w.textChanged.connect(self.change)
            
        # radio button parameters
        w_go_status = qt.QButtonGroup(self)
        w_go_status.setObjectName('go_status')
        w_go_status.setExclusive(True)
        but_run = qt.QRadioButton('RUN')
        but_pause = qt.QRadioButton('PAUSE')
        but_end = qt.QRadioButton('END')
        [b.setCheckable for b in [but_run, but_pause, but_end]]
        [w_go_status.addButton(b, i) for b,i in zip([but_run,but_pause,but_end],[self.RUN,self.PAUSE,self.END])]
        for w in [w_go_status]:
            w.buttonPressed.connect(self.change)

        # place widgets
        grid.addWidget(self.label1, 0,0)
        grid.addWidget(w_std_thresh,1,1)
        grid.addWidget(but_run,0,2)
        grid.addWidget(but_pause,0,3)
        grid.addWidget(but_end,0,4)

        # setup parameter holders
        self.param_names = ['std_thresh','trial_num','go_status','total_trials']
        defaults = [1.5,0,1,5]
        self.params = {n:d for n,d in zip(self.param_names,defaults)}
        
        self.setLayout(grid)
        self.setup_experiment()
    def setup_experiment(self):
        cam = Camera(idx=0, resolution=(320,240), frame_rate=50, color_mode=BW)
        trigger = Trigger(msg=[0,0,1,1], duration=5.0)
        daq = DAQ()
        self.exp = Experiment(camera=cam, daq=DAQ(), trigger=trigger, movement_std_thresh=self.params['std_thresh'], mask_names=['WHEEL', 'EYE'])
    def change(self):
        sender = self.sender()
        if isinstance(sender, qt.QLineEdit):
            self.params[sender.objectName()] = sender.text()
        elif isinstance(sender, qt.QButtonGroup):
            self.params[sender.objectName()] = sender.checkedId()
            if sender.objectName()=='go_status':
                print "doing something"
                if self.params['go_status']==self.RUN:
                    self.start_run()
                elif self.params['go_status']==self.END:
                    self.end()
    def start_run(self):
        self.exp.new_run(new_masks=False, trials=self.params['total_trials'])
        raw_input()
        #qt.QApplication.flush()
        #self.timer = qtc.QTimer()
        #timer.timeout.connect(self.exp.step)
        #self.timer.start(1)
    def end(self):
        self.exp.end()
        
class Interface(object):
    def __init__(self):
        
        self.app = qt.QApplication([])
        self.main = ExpWindow()
        self.main.show()

    def go(self):
        self.app.exec_()
        
