import flycapture2 as fc2
import numpy as np
import cv2
import sys

"""
NOTES:

all tests are working with regard to simple functions.
next big goal: enable software trigger so that memory and writing are not an issue at high framerate

interesting alternative API: https://bugs.launchpad.net/pydc1394/+bug/618004

useful links:
format7: http://cars.uchicago.edu/software/epics/PointGreyDoc.html
api: http://bytedeco.org/javacpp-presets/flycapture/apidocs/org/bytedeco/javacpp/FlyCapture2.Camera.html#GetFormat7Configuration-org.bytedeco.javacpp.FlyCapture2.Format7ImageSettings-int:A-float:A-
camera: http://www.ptgrey.com/support/downloads/10130
"""
def diff(ts1, ts2):
    return (ts2['cycle_secs']-ts1['cycle_secs']) + (ts2['cycle_count']-ts1['cycle_count'])/8000.

#print fc2.get_library_version()
c = fc2.Context()
#print c.get_num_of_cameras()
c.connect(*c.get_camera_from_index(0))
#print c.get_camera_info()
sys.exit(0)

#imset, packetsize, perc= c.get_format7_configuration()
#c.set_format7_configuration(imset, 100.) #100 makes it the fastest
#print imset

#vw = cv2.VideoWriter('test.avi', cv2.cv.CV_FOURCC('M','J','P','G'), 30, (imset['width'], imset['height']))
#if not vw.isOpened():
#    raise Exception('no write')

#c.set_video_mode_and_frame_rate(fc2.VIDEOMODE_800x600RGB ,fc2.FRAMERATE_60)
#m, f = c.get_video_mode_and_frame_rate()
#print m, f
#print c.get_video_mode_and_frame_rate_info(m, f)

#print c.get_property_info(fc2.FRAME_RATE)
#p = c.get_property(fc2.FRAME_RATE)
#print p
#c.set_property(**p)

c.set_timestamping(True)
c.start_capture()
im = fc2.Image()
last_ts = dict(cycle_secs=0,cycle_count=0)
while True:
    c.retrieve_buffer(im)
    img = np.array(im)
    ts = im.get_timestamp()
    cv2.imshow('a',img)
    print diff(last_ts,ts)
    cv2.waitKey(1)
    last_ts = ts
    #vw.write(img)
    
vw.release()
c.stop_capture()
c.disconnect()
cv2.destroyAllWindows()
