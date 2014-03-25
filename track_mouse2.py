import numpy as np
from matplotlib import path as mpl_path
import matplotlib.cm as mpl_cm
from progressbar import *
import sys
import pylab as pl
pl.ion()
import cv2
cv = cv2.cv
import os
import json

cv2.namedWindow('a')


thresh = 80
duration = 1
delta_max = 100
delta_min = 5

centers = []
left = 0
right = 0
skipped = 0
big_jumps = 0

def get_center(bounds):
    x = int(bounds[0]+round(0.5*bounds[2]))
    y = int(bounds[1]+round(0.5*bounds[3]))
    return np.array([x,y])
def dist(pt1, pt2):
    return np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)

try:
    moviedir = sys.argv[1]
except IndexError:
    print "Supply movie directory as argument."
    sys.exit(0)

startdir = os.getcwd()
os.chdir(moviedir)

timefile = [i for i in os.listdir('.') if 'timestamps' in i][0]
time = json.loads(open(timefile,'r').read())[0]
vidfile = [i for i in os.listdir('.') if '.avi' in i][0]

mov = cv2.VideoCapture(vidfile)
valid, frame = mov.read()
frame = cv2.cvtColor(frame, cv2.cv.CV_RGB2GRAY)
pl.imshow(frame, cmap=mpl_cm.Greys_r)
print "Select left room."
pts_l = pl.ginput(4)
print "Select right room."
pts_r = pl.ginput(4)
print "Select mouse."
pts_mouse = pl.ginput(4)
pl.close()
last_center = np.array([np.mean([p[0] for p in pts_mouse]), np.mean([p[1] for p in pts_mouse])], dtype=np.uint8)
path_l, path_r = [mpl_path.Path(pts) for pts in [pts_l,pts_r]]
allpoints = pts_l+pts_r
left_border = np.min([i[0] for i in allpoints])
right_border = np.max([i[0] for i in allpoints])
top_border = np.min([i[1] for i in allpoints])
bottom_border = np.max([i[1] for i in allpoints])

height, width = np.shape(frame)
motion_history = np.zeros((height, width), np.float32)
idx = 0
widgets=[' Iterating through images...', Percentage(), Bar()]
pbar = ProgressBar(widgets=widgets, maxval=len(time)).start()
while True:
    last = frame
    valid,frame = mov.read()
    if not valid:
        break
    frame = cv2.cvtColor(frame, cv2.cv.CV_RGB2GRAY)
    diff = cv2.absdiff(frame, last)
    _, motion_mask = cv2.threshold(diff, thresh, 1, cv2.THRESH_BINARY)
    cv2.updateMotionHistory(motion_mask, motion_history, idx, duration)
    gradient_mask, gradient_orientation = cv2.calcMotionGradient(motion_history, delta_max, delta_min)
    seg_mask, seg_bounds = cv2.segmentMotion(motion_history, idx, delta_max)
    seg_bounds = [b for b in seg_bounds if  get_center(b)[0] > left_border and get_center(b)[0] < right_border and get_center(b)[1]>top_border and get_center(b)[1]<bottom_border]
    if len(seg_bounds):
        dx = [dist(get_center(b),last_center) for b in seg_bounds]
        sz = [b[2]*b[3] for b in seg_bounds]
        chosen_idx = np.argmax(sz)#np.argmin(dx)
        chosen = seg_bounds[chosen_idx]
        
        center = get_center(chosen)
    else:
        #print "skipped a frame because no difference"
        center = last_center
        skipped += 1
    
    if dist(center, last_center) > 200:
        big_jumps += 1
    last_center = center
    centers.append(center)

    showimg = np.copy(frame)
    if path_l.contains_point(center):
        left+=1
        cv2.circle(showimg, tuple(center), radius=10, color=(20,20,20))
    if path_r.contains_point(center):
        right+=1
        cv2.circle(showimg, tuple(center), radius=3, color=(20,20,20))
    cv2.imshow('a',showimg)
    cv2.waitKey(1)
    idx += 1
    pbar.update(idx)
pbar.finish()
disp_img = last
[cv2.circle(disp_img, tuple(c), radius=3, color=(20,20,20)) for c in centers]
cv2.imwrite(moviedir+'_summary.png',disp_img)

os.chdir(startdir)
mov.release()
cv2.destroyWindow('a')


