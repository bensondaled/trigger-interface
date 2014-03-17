from monitor import Monitor
from cameras import Camera, BW, COLOR
import numpy as np
from playback import Playback as PB
import os

name = None
while name==None or name in os.listdir('.'):
    name = raw_input('Enter name of experiment (will default to date string):')
    if name in os.listdir('.'):
        print "Experiment file with this name already exists."

duration = None
while type(duration) != float:
    try:
        duration = float(raw_input('Enter duration of experiment in seconds:'))
    except:
        pass

cam = Camera(0, frame_rate=30, color_mode=BW)
mon = Monitor(cam, show=True, run_name=name, duration=duration)
mon.go()
