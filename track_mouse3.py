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

def get_frame(mov, n=1):
    kernel = 9
    valid,frame = mov.read()
    if valid:
        frame = frame.astype(np.float32)
        frame = cv2.cvtColor(frame, cv2.cv.CV_RGB2GRAY)
        frame = cv2.GaussianBlur(frame, (kernel,kernel), 0)
    for i in range(n-1):
        valid,new = mov.read()
        if valid:
            new = new.astype(np.float32)
            new = cv2.cvtColor(new, cv2.cv.CV_RGB2GRAY)
            new = cv2.GaussianBlur(new, (kernel,kernel), 0)
        frame = frame+new
    if valid:
        frame = frame/n
    return (valid, frame)

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
get_frame(mov,n=1000) #for hands
valid, baseline = get_frame(mov, n=200)
height, width = np.shape(baseline)
pl.imshow(baseline, cmap=mpl_cm.Greys_r)
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

idx = 0
widgets=[' Iterating through images...', Percentage(), Bar()]
pbar = ProgressBar(widgets=widgets, maxval=len(time)).start()
while True:
    valid,frame = get_frame(mov)
    if not valid:
        break
    diff = cv2.absdiff(frame,baseline).astype(np.uint8)
    edges = cv2.Canny(diff, 10, 100)
    contours, hier = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    
    #centers.append(center)
    
    #display
    showimg = np.copy(frame).astype(np.uint8)
#    if path_l.contains_point(center):
#        left+=1
#        cv2.circle(showimg, tuple(center), radius=10, thickness=2, color=(255,255,255))
#    if path_r.contains_point(center):
#        right+=1
#        cv2.circle(showimg, tuple(center), radius=10, thickness=3,color=(255,255,255))
#    cv2.circle(showimg, tuple(center), radius=10, thickness=5, color=(255,255,255))
#    if len(possible):
#        for q in np.argsort(sz)[-10:]:
#            b = seg_bounds[q]
#            pt1 = (b[0],b[1])
#            pt2 = (b[0]+b[2], b[1]+b[3])
#            cv2.rectangle(showimg,pt1,pt2,20) 
    cv2.imshow('img',showimg)
    cv2.drawContours(diff,contours,-1,(0,255,0),3)
    cv2.imshow('diff', diff)
    cv2.waitKey(1)
    #/display

    idx += 1
    #last_center = center
    pbar.update(idx)
pbar.finish()
#disp_img = last
#cv2.circle(disp_img, tuple(c), radius=3, color=(20,20,20)) for c in centers]
#cv2.imwrite(moviedir+'_summary.png',disp_img)

os.chdir(startdir)
mov.release()
cv2.destroyWindow('a')


