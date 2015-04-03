import numpy as np
import pylab as pl
import cv2
import sys,os
pjoin = os.path.join
from mapping_playback import FileHandler

if __name__ == '__main__':
    data_dir = pjoin('.','data')
    name = ''
    fh = FileHandler(data_dir, name)
    for ti in xrange(1,fh.n_trials+1):
        ts = np.load(fh.get_path(ti, fh.TIME))
        nframes = []
        nts = []
        for ci in xrange(1,fh.n_cams+1):
            c = cv2.VideoCapture(fh.get_path(ti, fh.CAM, ci))
            nframes.append(c.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
            nts.append(ts['time%i'%ci].shape[0])
        nframes = np.array(nframes)
        nts = np.array(nts)
        print np.all(nframes==nts)

    
