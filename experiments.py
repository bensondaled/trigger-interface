import numpy as np 
import os
import cv2
cv = cv2.cv
import time
import pylab as pl
import json
import matplotlib.cm as mpl_cm
from matplotlib import path as mpl_path
import matplotlib.animation as ani
from daq import DAQ, Trigger
from cameras import Camera

NP = 0
CV = 1

class Experiment(object):
	def __init__(self, cameras=None, daq=None, save_mode=NP, mask_names=('WHEEL','EYE'), monitor_cam_idx=0, motion_mask='WHEEL', movement_query_frames=20, movement_std_thresh=2.0, trigger=None, inter_trial_min=5.0):
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
		self.monitor_img_sets = [self.empty_img_set(i) for i,cam in enumerate(self.cameras)]
				
		self.mask_names = mask_names
		self.masks = {}
		self.mask_idxs = {}
		self.motion_mask = motion_mask
		
		self.windows = self.make_windows()
		
		self.movement_std_thresh = movement_std_thresh
		self.movement_query_frames = movement_query_frames
		
		self.trigger = trigger
		self.inter_trial_min = inter_trial_min
		
		self.TRIAL_ON = False
		self.last_trial_off = time.time()
		
		self.run_info = self.empty_run_info()
		
		self.writers = []
	def metadata(self):
		md = {}
		md['movement_std_thresh'] = self.movement_std_thresh
		md['movement_query_frames'] = self.movement_query_frames
		md['n_cameras'] = len(self.cameras)
		md['monitor_cam_idx'] = self.monitor_cam_idx
		
		return md
	def empty_run_info(self):
		runinf = {}
		runinf['trigger_times'] = []
		return runinf
	def new_run(self, new_masks=False,trials=None):		
		if new_masks or len(self.masks)==0:	
			self.set_masks()
		self.trials_total = trials
		self.trial_count = 0

		self.img_sets = [self.empty_img_set(cam_idx) for cam_idx in range(len(self.cameras))]
		self.times = [[] for i in self.cameras]	

		self.run_name = time.strftime("%Y%m%d_%H%M%S")
		
		self.run_info = self.empty_run_info()
		os.mkdir(self.run_name)
		os.chdir(self.run_name)
		
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
	def end_run(self):
		f = open("%s-data.json"%self.run_name, 'w')
		f.write("%s"%json.dumps(self.run_info))
		f.close()
		os.chdir('..')
	def empty_img_set(self, cam_idx):
		return None #np.array([np.empty(self.cameras[cam_idx].resolution[::-1])])
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
			self.mask_idxs[m] = np.where(mask==False)
	def save(self, cam_idx=None, frame=None):
		if self.save_mode == CV:
			self.writers[cam_idx].write(frame)
		elif self.save_mode == NP:
			for cam_idx in range(len(self.cameras)):
				np.savez_compressed(self.run_name+'-cam%i-trial%i'%(cam_idx,self.trial_count), time=self.times[cam_idx], data=self.img_sets[cam_idx])
				self.img_sets[cam_idx] = self.empty_img_set(cam_idx)
				self.times[cam_idx] = []
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
		if time.time()-self.last_trial_off < self.inter_trial_min:
			return False
		mask_idxs = self.mask_idxs[self.motion_mask]
		std_pts = np.std(self.monitor_img_sets[self.monitor_cam_idx][:,mask_idxs[0],mask_idxs[1]], axis=0)
		return np.mean(std_pts) < self.movement_std_thresh
	def record_frame(self, frame, cam_idx):
		if self.img_sets[cam_idx] != None:
			self.img_sets[cam_idx] = np.append(self.img_sets[cam_idx], [frame], axis=0)
		else:
			self.img_sets[cam_idx] = np.array([frame])
	def monitor_frame(self, frame, cam_idx):
		if self.monitor_img_sets[cam_idx] != None:
			self.monitor_img_sets[cam_idx] = np.append(self.monitor_img_sets[cam_idx], [frame], axis=0)
		else:
			self.monitor_img_sets[cam_idx] = np.array([frame])
		if np.shape(self.monitor_img_sets[cam_idx])[0]>self.movement_query_frames:
			self.monitor_img_sets[cam_idx] = self.monitor_img_sets[cam_idx][-self.movement_query_frames:]
	def next_frame(self):
		for cam_idx,win,cam in zip(range(len(self.cameras)),self.windows,self.cameras):
			frame = cam.read()
			if self.TRIAL_ON:
				self.record_frame(frame, cam_idx)
				self.times[cam_idx].append(time.time())
			elif not self.TRIAL_ON:
				if self.save_mode == CV:	self.save(cam_idx, frame)
				self.monitor_frame(frame, cam_idx)
				cv2.imshow(win, frame)
	def send_trigger(self):
		self.daq.trigger(self.trigger)
		self.run_info['trigger_times'].append(time.time())
	def run(self, **kwargs):
		# preparations for new run
		self.new_run(**kwargs)
		
		# run first few frames before any checking
		for i in range(self.movement_query_frames):
			self.next_frame()
		
		# main loop
		while True:
			self.next_frame()
			
			if self.TRIAL_ON:
				if time.time()-self.TRIAL_ON >= self.trigger.duration:
					self.TRIAL_ON = False
					self.last_trial_off = time.time()
					if self.save_mode == NP:	self.save()
			
			if not self.TRIAL_ON:		
				c = cv2.waitKey(1)
				if c == ord('q') or (self.trials_total and self.trial_count==self.trials_total):
					break
					
				if self.query_for_trigger():
					self.monitor_img_sets[self.monitor_cam_idx] = self.empty_img_set(self.monitor_cam_idx)
					self.send_trigger()
					self.TRIAL_ON = time.time()
					self.trial_count += 1
					
		self.end_run()
		
	

	
