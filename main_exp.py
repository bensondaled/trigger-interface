from core.cameras import Camera
from core.daq import Trigger
from imaging_interface import Experiment, TriggerCycle

cam =  Camera(idx=0, resolution=(320,240), frame_rate=200, color_mode=Camera.BW)

CS = Trigger(msg=[0,0,1,1], duration=5.0, name='CS')
US = Trigger(msg=[0,0,0,1], duration=5.0, name='US')
trigger_cycle = TriggerCycle(triggers=[CS, US, CS, CS])
        
exp = Experiment(camera=cam, trigger_cycle=trigger_cycle, n_trials=20, resample=1, movement_std_thresh=10, movement_query_frames=3)
exp.run() #'q' can always be used to end the run early. don't kill the process

#no buffer for eyelid
