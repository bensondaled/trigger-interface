from core.cameras import Camera
from core.daq import Trigger
from imaging_interface2 import Experiment
import os

data_dir = os.path.join('.','data')
name = ''
while name == '':
    name = raw_input('Enter experiment name:')
cam1 = Camera(cam_type=Camera.PG, resolution=(1280,960))
cam2 = Camera(idx=1, resolution=(320,240), frame_rate=40, color_mode=Camera.BW)

trig = Trigger(msg=[1,1,1,1], duration=1.0, name='basic')

exp = Experiment(name=name, camera1=cam1, camera2=cam2, data_dir=data_dir, trigger=trig)
exp.run() #'q' can always be used to end the run early. don't kill the process