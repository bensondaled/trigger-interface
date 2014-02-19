import pyDAQmx as pydaq
import numpy as np 
import cv2
import cv2.cv as cv
import time
import pylab as pl
import json
import matplotlib.cm as mpl_cm
from matplotlib import path as mpl_path
import matplotlib.animation as ani
pl.ion()

BW = 0
COLOR = 1

NP = 0
CV = 1

class Trigger(object):
	def __init__(self, msg=[], duration=1.0):
		self.duration = duration
		self._msg = None
		self.msg = msg
	
	@property
	def msg(self):
		return self._msg
	@msg.setter
	def msg(self, msg):
		self._msg = np.array(msg).astype(np.uint8)
		
	def metadata(self):
		md = {}
		md['duration'] = self.duration
		md['msg'] = str(self.msg)
		
		return md
class DAQ(object):
	def __init__(self, port="Dev1/Port1/Line0:3"):
		self.port = port
		self.task = pydaq.TaskHandle()
		pydaq.DAQmxCreateTask("", pydaq.byref(self.task))
		pydaq.DAQmxCreateDOChan(self.task, self.port, "OutputOnly", pydaq.DAQmx_Val_ChanForAllLines)
		pydaq.DAQmxStartTask(self.task)
	def trigger(self, trig):
		DAQmxWriteDigitalLines(self.task,1,1,10.0,pydaq.DAQmx_Val_GroupByChannel,trig.msg,None,None)
	def release(self):
		pydaq.DAQmxStopTask(self.task)
        pydaq.DAQmxClearTask(self.task)
		
	def metadata(self):
		md = {}
		md['port'] = self.port
		
		return md
class Camera(object):
	def __init__(self, idx=0, resolution=(320,240), frame_rate=50, color_mode=BW):
		self.resolution = resolution
		self.frame_rate = frame_rate
		self.color_mode = color_mode
		
		self.vc = cv2.VideoCapture(idx)

		self.vc.set(cv.CV_CAP_PROP_FPS, self.frame_rate)
		self.vc.set(cv.CV_CAP_PROP_FRAME_WIDTH, self.resolution[0])
		self.vc.set(cv.CV_CAP_PROP_FRAME_HEIGHT, self.resolution[1])
			
		time.sleep(0.1)
		self.vc.read()
	def read(self):
		success,frame = self.vc.read()
		if self.color_mode==BW:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		return frame
	def release(self):
		self.vc.release()
	def metadata(self):
		md = {}
		md['resolution'] = self.resolution
		md['frame_rate'] = self.frame_rate
		md['color_mode'] = self.color_mode
		
		return md
class Experiment(object):
	def __init__(self, cameras=None, daq=None, save_mode=NP, mask_names=('WHEEL','EYE'), monitor_cam_idx=0, motion_mask='WHEEL', movement_query_frames=20, movement_std_thresh=40, trigger=None):
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
		if type(cameras) == Camera:
			self.cameras = [cameras]
		elif type(cameras) == list:
			self.cameras = cameras
		elif cameras == None:
			self.cameras = []
		self.monitor_cam_idx = monitor_cam_idx
		
		self.daq = daq
		
		self.save_mode = save_mode
		self.img_sets = [self.empty_img_set(i) for i,cam in enumerate(self.cameras)]
				
		self.mask_names = mask_names
		self.masks = {}
		self.mask_idxs = {}
		self.motion_mask = motion_mask
		
		self.windows = self.make_windows()
		
		self.trial_count = 0
		self.movement_std_thresh = movement_std_thresh
		self.movement_query_frames = movement_query_frames
		
		self.trigger = trigger
		
		self.TRIAL_ON = False
	def metadata(self):
		md = {}
		md['movement_std_thresh'] = self.movement_std_thresh
		md['movement_query_frames'] = self.movement_query_frames
		md['n_cameras'] = len(self.cameras)
		md['monitor_cam_idx'] = self.monitor_cam_idx
		
		return md
	def new_run(self, new_masks=False):		
		if new_masks or len(self.masks)==0:	
			self.set_masks()
		
		self.run_name = time.strftime("%Y%m%d_%H%M%S")
		
		if self.save_mode == CV:
			[writer.release() for writer in self.writers]
			self.writers = [cv2.VideoWriter(self.run_name+"-cam%i.avi"%i,0,\
			cam.frame_rate,\
			frameSize=cam.resolution,\
			isColor=False) \
			for i,cam in enumerate(self.cameras)]
		
		dic = {}
		dic['run_name'] = self.run_name
		dic['experiment'] = self.metadata()
		dic['cameras'] = [cam.metadata() for cam in self.cameras]
		dic['daq'] = self.daq.metadata()
		dic['trigger'] = self.trigger.metadata()
		
		f = open("%s-metadata.json"%self.run_name, 'w')
		f.write("%s"%json.dumps(dic))
		f.close()
		
	def empty_img_set(self, cam_idx):
		return np.empty(self.cameras[cam_idx].resolution[::-1])
	def make_mask(self, cam_idx):
		frame = self.cameras[cam_idx].read()
		pl.imshow(frame, cmap=mpl_cm.Greys_r)
		pts = pl.ginput(0)
		pl.close()
		path = mpl_path.Path(pts)
		mask = np.ones(np.shape(frame), dtype=bool)
		for ridx,row in enumerate(mask):
			for cidx,pt in enumerate(row):
				if path.contains_point([cidx, ridx]):
					mask[ridx,cidx] = False
		return mask
	def set_masks(self, cam_idx=None):
		if cam_idx == None:
			cam_idx = self.monitor_cam_idx
		for m in self.mask_names:
			print "Please select mask: %s."%m
			mask = self.make_mask(cam_idx)
			self.masks[m] = mask
			self.mask_idxs[m] = np.nonzero(mask)
	def save(self, cam_idx=None, frame=None):
		if self.save_mode == CV:
			self.writers[cam_idx].write(frame)
		elif self.save_mode == NP:
			for cam_idx in range(len(self.cameras)):
				np.save(self.run_name+'-cam%i-trial%i'%(cam_idx,self.trial_count), self.img_sets[cam_idx])
				self.img_sets[cam_idx] = empty_img_set(cam_idx)
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
		if self.save_mode == CV:
			[writer.release() for writer in self.writers]
		self.daq.release()
	def query_for_trigger(self):
		frames = self.img_sets[self.monitor_cam_idx][-self.movement_query_frames:] #[:,:,-self.movement_query_frames:]
		#method 1:
		mask_idxs = self.mask_idxs[self.motion_mask]
		std_pts = np.std(frames[:,mask_idxs[0],mask_idxs[1]], axis=0)
		return np.mean(std_pts) < self.movement_std_thresh
		
		#could also try multiplying frames by the mask (not mask_idxs)
		
		#could also try using flat idxs (use flatnonzero when originally creating the mask) and apply those to frames.flat somehow
	def next_frame(self):
		for cam_idx,win,cam in zip(range(len(self.cameras)),self.windows,self.cameras):
			frame = cam.read()
			self.img_sets[cam_idx] = np.append(self.img_sets[cam_idx], [frame], axis=0) #np.dstack((self.img_sets[cam_idx],frame))

			if self.save_mode == CV:	self.save(cam_idx, frame)
			if not self.TRIAL_ON:
				cv2.imshow(win, frame)
	def run(self):
		# preparations for new run
		self.new_run()
		
		# run first few frames before any checking
		for i in range(self.movement_query_frames):
			self.next_frame()
		
		# main loop
		while True:
			self.next_frame()
			
			if self.TRIAL_ON:
				if time.time()-self.TRIAL_ON >= self.trigger.duration:
					self.TRIAL_ON = False
					if self.save_mode == NP:	self.save()
					self.trial_count += 1
			
			if not self.TRIAL_ON:		
				c = cv2.waitKey(1)
				if c == ord('q'):
					break
					
				if self.query_for_trigger():
					self.daq.trigger(self.trigger)
					self.TRIAL_ON = time.time()
		
	
if __name__=='__main__':
	monitor_cam = Camera(idx=0, resolution=(320,240), frame_rate=50, color_mode=BW)
	behaviour_cam = Camera(idx=1, resolution=(160, 120), frame_rate=10, color_mode=BW)
	trigger = Trigger(msg=[0,0,1,1], duration=5.0)
	
	exp = Experiment(cameras=[monitor_cam, behaviour_cam], monitor_cam_idx=0, save_mode=NP, trigger=trigger)
	
	#exp.run(new_masks=True)
	#exp.end()



	