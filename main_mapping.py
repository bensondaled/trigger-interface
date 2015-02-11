from core.cameras import Camera
from core.daq import Trigger
from imaging_interface2 import Experiment
import os

data_dir = os.path.join('.','data')
name = ''
while name == '':
    name = raw_input('Enter experiment name:')
    
cam1 = Camera(cam_type=Camera.PG, resolution=(688,504)) #resolution for PG is not a choice but a relay
cam2 = Camera(idx=1, cam_type=Camera.PSEYE, resolution=(320,240), frame_rate=85, color_mode=Camera.COLOR)

trig = Trigger(msg=[1,1,1,1], name='basic')
trial_duration = 3.0 #seconds
trigger_delay = 1.0 #seconds

exp = Experiment(name=name, camera1=cam1, camera2=cam2, data_dir=data_dir, trigger=trig, trial_duration=trial_duration, stim_delay=trigger_delay)
exp.run() #'q' is used to end. Do not kill process.
