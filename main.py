from playback import Playback
import cameras
from daq import Trigger, DAQ
import experiments as exp

monitor_cam =  cameras.Camera(idx=0, resolution=(320,240), frame_rate=50, color_mode=cameras.BW)
behaviour_cam = cameras.Camera(idx=1, resolution=(160, 120), frame_rate=10, color_mode=cameras.BW)
trigger = Trigger(msg=[0,0,1,1], duration=5.0)
        
exp = exp.Experiment(cameras=[monitor_cam], daq=DAQ(), monitor_cam_idx=0, save_mode=exp.NP, trigger=trigger)
        
exp.run(new_masks=True, trials=3)
exp.end()

