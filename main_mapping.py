from core.cameras import Camera
from core.daq import Trigger
from interface_1p import Experiment
import os

data_dir = os.path.join('.','data')
name = ''
while name == '':
    name = raw_input('Enter experiment name:').lower()
    
pseye_params = dict(gain = 80,\
exposure = 34,\
wbal_red = 50,\
wbal_blue = 50,\
wbal_green = 50,\
vflip = True)

pseye_params2 = dict(gain = 50,\
exposure = 100,\
wbal_red = 50,\
wbal_blue = 50,\
wbal_green = 50,\
vflip = False)

cam1 = Camera(cam_type=Camera.PG, resolution=(1280,960)) #resolution for PG is not a choice, rather it must match PG
#cam2 = Camera(idx=1, cam_type=Camera.PSEYE, resolution=(320,240), frame_rate=85, color_mode=Camera.COLOR)
#cam2 = Camera(idx=1-1, cam_type=Camera.PSEYE_NEW, resolution=(320,240), frame_rate=125, **pseye_params2)

#frontal view
#cam2 = Camera(idx=1-1, cam_type=Camera.PSEYE_NEW, resolution=(640,480), frame_rate=75, **pseye_params2)
#cam3 = Camera(idx=2-1, cam_type=Camera.PSEYE_NEW, resolution=(320,240), frame_rate=125, **pseye_params)
#lateral view
pseye_params['vflip']=False;
pseye_params2['vflip']=True;

cam2 = Camera(idx=1-1, cam_type=Camera.PSEYE_NEW, resolution=(320,240), frame_rate=125, **pseye_params)
cam3 = Camera(idx=2-1, cam_type=Camera.PSEYE_NEW, resolution=(640,480), frame_rate=75, **pseye_params2)


trig = Trigger(msg=[1,1,1,1], name='basic')
trial_duration = 5.0 #seconds
trigger_delay = 2.0 #seconds

exp = Experiment(name=name, camera1=cam1, camera2=cam2, camera3=cam3, data_dir=data_dir, trigger=trig, trial_duration=trial_duration, stim_delay=trigger_delay)
exp.run() #'q' is used to end. Do not kill process.

# PSEye frame rates: 15, 30, 60, 75, 100, 125