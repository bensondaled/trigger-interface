#from cameras import Camera, BW, COLOR
#from daq import Trigger, DAQ
#from andrea import Experiment
#from playback import Playback
#
#cam =  Camera(idx=0, resolution=(320,240), frame_rate=50, color_mode=BW)
#trigger = Trigger(msg=[0,0,1,1], duration=5.0)
#daq = DAQ()
#
#thresh = 1.5
#
#exp = Experiment(camera=cam, daq=DAQ(), trigger=trigger, movement_std_thresh=thresh, mask_names=['WHEEL', 'EYE'])
#
#exp.run(new_masks=False, trials=5) #'q' can always be used to end the run early. don't kill the process
#exp.end()

from andrea import Interface
i = Interface()
