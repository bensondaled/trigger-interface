from cameras import Camera, BW, COLOR
from daq import Trigger, DAQ
from experiments import Experiment, NP, CV
from playback import Playback

monitor_cam =  Camera(idx=0, resolution=(320,240), frame_rate=50, color_mode=BW)
behaviour_cam = Camera(idx=1, resolution=(160, 120), frame_rate=10, color_mode=BW)
trigger = Trigger(msg=[0,0,1,1], duration=5.0)
        
exp = Experiment(cameras=[monitor_cam], daq=DAQ(), monitor_cam_idx=0, save_mode=NP, trigger=trigger)
        
exp.run(new_masks=False, trials=3) #'q' can always be used to end the run early. don't kill the process
#exp.run(new_masks=False, trials=10)
#exp.end()

