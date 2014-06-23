from core.cameras import Camera
from core.daq import Trigger
from imaging_interface import Experiment, TriggerCycle

#name = 'AG052014-02/062214_-191 -226 72_WHISK_COND_'
name = 'AG051514-01/062214_-34 490 193_EXT_A_'
#name = 'AG052914-03/061714_UNKNOWN LOOK STACK AND PICTURE_COND_A'

#name = 'AG052914-01/061914__114 -328 8_US_ONLY'

cam =  Camera(idx=0, resolution=(320,240), frame_rate=200, color_mode=Camera.BW)

CS = Trigger(msg=[0,0,1,1], duration=4.0, name='CS')
US = Trigger(msg=[1,1,0,0], duration=4.0, name='US')
CSUS = Trigger(msg=[1,1,1,1], duration=4.0, name='CSUS')


#trigger_cycle = TriggerCycle(triggers=[CSUS, CSUS, CSUS, CSUS, CS, CSUS, CSUS, CSUS, CSUS, US])
#trigger_cycle = TriggerCycle(triggers=[CSUS, CSUS, CSUS, CSUS, CSUS, CSUS, CSUS, CSUS,CS, CSUS, CSUS, CSUS, CSUS, CSUS, CSUS, CSUS, CSUS, CSUS, CSUS,US])
#trigger_cycle = TriggerCycle(triggers=[CS, CS, CS])


#trigger_cycle = TriggerCycle(triggers=[CSUS, CSUS, CSUS, CSUS, CSUS, CSUS, CSUS, CSUS])
#trigger_cycle = TriggerCycle(triggers=[US, US, US, US, US, US, US, US])
#trigger_cycle = TriggerCycle(triggers=[CS, CS, CS, CS, CS, CS, CS, CS])
trigger_cycle = TriggerCycle(triggers=[CS, CS, CS, US, CS, CS, CS, US, CS, CS])


exp = Experiment(name=name, camera=cam, trigger_cycle=trigger_cycle, n_trials=-1)
exp.run() #'q' can always be used to end the run early. don't kill the process

"""
# Important parameters for Experiment object:

camera: a Camera object, see examples
trigger_cycle: the triggers of the experiment, see examples

movement_query_frames: the number of frames in the buffer that calculates wheel movement
inter_trial_min: minimum number of seconds between triggers
n_trials: number of triggers (give -1 for limitless)
resample: look at every n'th frame when performing wheel and eyelid calculations
monitor_vals_display: number of values (eyelid mean and wheel std) to show in window
"""
