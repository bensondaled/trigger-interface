from monitor import Monitor
from cameras import Camera, BW, COLOR
import numpy as np
from playback import Playback as PB

cam = Camera(0, frame_rate=10, color_mode=BW)
mon = Monitor(cam, show=True)
mon.run(duration=60)

