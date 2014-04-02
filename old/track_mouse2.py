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

cv2.namedWindow('img')
cv2.namedWindow('diff')
cv2.namedWindow('motionmask')


thresh = 15
duration = 1
seg_thresh = 100
translation_thresh = 50

centers = []
left = 0
right = 0
skipped = 0

def ginput(n):
    pts = pl.ginput(n)
    pts = np.array(pts)
    #for idx,p in enumerate(pts):
    #   pts[idx][0] = height-p[0]
    return pts

def get_frame(mov,first=False, acc=None):
    valid,frame = mov.read()
    if valid:
        frame = frame.astype(np.float32)
        frame = cv2.cvtColor(frame, cv2.cv.CV_RGB2GRAY)
        frame = cv2.GaussianBlur(frame, (7,7), 0)
        if first:
            acc = np.copy(frame)
        elif not first:
            cv2.accumulateWeighted(np.copy(frame), acc, 0.2)
    return (valid, frame, acc)

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
for i in range(550):
    mov.read()
valid, frame, acc = get_frame(mov, first=True)
height, width = np.shape(frame)
pl.imshow(frame, cmap=mpl_cm.Greys_r)
print "Select left room."
pts_l = ginput(4)
print "Select right room."
pts_r = ginput(4)
print "Select mouse."
pts_mouse = ginput(4)
pl.close()
last_center = np.round(np.mean(pts_mouse, axis=0)).astype(int)
path_l, path_r = [mpl_path.Path(pts) for pts in [pts_l,pts_r]]
allpoints = pts_l+pts_r
left_border = np.min([i[0] for i in allpoints])
right_border = np.max([i[0] for i in allpoints])
top_border = np.min([i[1] for i in allpoints])
bottom_border = np.max([i[1] for i in allpoints])

motion_history = np.zeros((height, width), np.float32)
idx = 0
widgets=[' Iterating through images...', Percentage(), Bar()]
pbar = ProgressBar(widgets=widgets, maxval=len(time)).start()
while True:
    last = frame
    valid,frame,acc = get_frame(mov, acc=acc)
    if not valid:
        break
    diff = cv2.absdiff(frame,acc) #use last, or use acc instead of last for an averaged "last frame"
    _, motion_mask = cv2.threshold(diff, thresh, 1, cv2.THRESH_BINARY)
    cv2.updateMotionHistory(motion_mask.astype(np.uint8), motion_history, idx, duration)
    seg_mask, seg_bounds = cv2.segmentMotion(motion_history, idx, seg_thresh)
    seg_bounds = [b for b in seg_bounds if  get_center(b)[0] > left_border and get_center(b)[0] < right_border and get_center(b)[1]>top_border and get_center(b)[1]<bottom_border]
    dx = [dist(get_center(b),last_center) for b in seg_bounds]
    possible = [seg_bounds[b] for b in np.argsort(dx) if dx[b]<translation_thresh ]
    if len(possible):
        sz = [b[2]*b[3] for b in possible]
        chosen_idx = np.argmax(sz)#np.argmin(dx)
        chosen = seg_bounds[chosen_idx]
        center = get_center(chosen)
    else:
        center = last_center
        skipped += 1
    
    centers.append(center)
    
    #display
    showimg = np.copy(frame).astype(np.uint8)
    if path_l.contains_point(center):
        left+=1
        cv2.circle(showimg, tuple(center), radius=10, thickness=2, color=(255,255,255))
    if path_r.contains_point(center):
        right+=1
        cv2.circle(showimg, tuple(center), radius=10, thickness=3,color=(255,255,255))
    cv2.circle(showimg, tuple(center), radius=10, thickness=5, color=(255,255,255))
    if len(possible):
        for q in np.argsort(sz)[-10:]:
            b = seg_bounds[q]
            pt1 = (b[0],b[1])
            pt2 = (b[0]+b[2], b[1]+b[3])
            cv2.rectangle(showimg,pt1,pt2,20) 
    cv2.imshow('img',showimg)
    cv2.imshow('diff', diff)
    cv2.imshow('motionmask',motion_mask)
    cv2.waitKey(1)
    #/display

    idx += 1
    last_center = center
    pbar.update(idx)
pbar.finish()
disp_img = last
[cv2.circle(disp_img, tuple(c), radius=3, color=(20,20,20)) for c in centers]
cv2.imwrite(moviedir+'_summary.png',disp_img)

os.chdir(startdir)
mov.release()
cv2.destroyWindow('a')


