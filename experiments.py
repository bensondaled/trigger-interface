import pyDAQmx as pydaq
import numpy as np 
import cv2
import cv2.cv as cv
import time
import pylab as pl
import matplotlib.cm as mpl_cm
from matplotlib import path as mpl_path
import matplotlib.animation as ani
pl.ion()

BW = 0
COLOR = 1

NP = 0
CV = 1

class DAQ(object):
	def __init__(self):
		self.task = pydaq.TaskHandle()
		pydaq.DAQmxCreateTask("", byref(self.task))
		pydaq.DAQmxCreateDOChan(self.task,"Dev1/Port1/Line0:3","OutputOnly",pydaq.DAQmx_Val_ChanForAllLines)
		pydaq.DAQmxStartTask(self.task)
	def trigger(self, msg):
		msg = np.array(msg).astype(np.uint8)
		DAQmxWriteDigitalLines(self.task,1,1,10.0,pydaq.DAQmx_Val_GroupByChannel,msg,None,None)
	def release(self):
		pydaq.DAQmxStopTask(self.task)
        pydaq.DAQmxClearTask(self.task)
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
class Experiment(object):
	def __init__(self, cameras=None, daq=None, save_mode=NP, mask_names=('WHEEL','EYE'), monitor_cam_idx=0, motion_mask='WHEEL', movement_query_frames=20, movement_std_thresh=40, daq_msg=[]):
		self.name = time.strftime("%Y%m%d_%H%M%S")

		if type(cameras) == Camera:
			self.cameras = [cameras]
		elif type(cameras) == list:
			self.cameras = cameras
		elif cameras == None:
			self.cameras = []
		self.monitor_cam_idx = monitor_cam_idx
		
		self.daq = daq
		
		self.save_mode = save_mode
		if self.save_mode == CV:
			self.writers = [cv2.VideoWriter(self.name+"-cam%i.avi"%i,0,\
			cam.frame_rate,\
			frameSize=cam.resolution,\
			isColor=False) \
			for i,cam in enumerate(self.cameras)]
		self.img_sets = [np.empty(cam.resolution[::-1]) for cam in self.cameras]

		
		self.mask_names = mask_names
		self.masks = {}
		self.mask_idxs = {}
		self.motion_mask = motion_mask
		
		self.windows = self.make_windows()
		
		self.trial_count = 0
		self.movement_std_thresh = movement_std_thresh
		self.movement_query_frames = movement_query_frames
		
		self.daq_msg = daq_msg
		
		self.TRIAL_ON = False
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
	def save(self, cam_idx, frame):
		if self.save_mode == CV:
			self.writers[cam_idx].write(frame)
		elif self.save_mode == NP:
			np.save(self.name+'-cam%i-trial%i'%(cam_idx,self.trial_count), self.img_sets[cam_idx])
	def make_windows(self):
		windows = [str(i) for i in range(len(self.cameras))]
		for w in windows:
			cv2.namedWindow(w, cv.CV_WINDOW_NORMAL)
		return windows
	def add_camera(self, cam):
		self.cameras.append(cam)
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
		# get and set masks
		self.set_masks()
		
		# run first few frames before any checking
		for i in range(self.movement_query_frames):
			self.next_frame()
		
		# main loop
		while True:
			self.next_frame()
			
			if not self.TRIAL_ON:		
				if self.query_for_trigger():
					self.daq.trigger(self.daq_msg)
					self.TRIAL_ON = True
			
				c = cv2.waitKey(1)
				if c == ord('q'):
					break
		
	
if __name__=='__main__':
	exp = Experiment(cameras=Camera(), save_mode=CV)
	
	exp.run()
	exp.end()



	