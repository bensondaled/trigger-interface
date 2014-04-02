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

show = raw_input('Show? (0/1) ')
show = int(show)

SHOW = show
RESAMPLE = 8

cv2.namedWindow('img')
cv2.namedWindow('diff')


thresh = 15
duration = 1
seg_thresh = 100
translation_thresh = 130
cth1 = 0 #canny thresh
cth2 = 0
diff_thresh = 100
kernel = 15

centers = []
track = None
left = 0
right = 0
left_assumed = 0
right_assumed = 0
skipped = 0

def ginput(n):
    pts = pl.ginput(n)
    pts = np.array(pts)
    #for idx,p in enumerate(pts):
    #   pts[idx][0] = height-p[0]
    return pts

def get_frame(mov, n=1, skip=0, blur=True):
    for s in range(skip):
        mov.read()
    if n==-1:
        n = 99999999999999999.
    def get():
        valid, frame = mov.read()
        if not valid:
            return (False, None)
        frame = frame.astype(np.float32)
        frame = cv2.cvtColor(frame, cv2.cv.CV_RGB2GRAY)
        if blur:
            frame = cv2.GaussianBlur(frame, (kernel,kernel), 0)
        return valid,frame

    valid,frame = get()
    i = 1
    while valid and i<n:
        valid,new = get()
        i += 1
        if valid:
            frame += new
    
    if frame!=None:
        frame = frame/i
    return (valid, frame)

def contour_center(c):
    return np.round(np.mean(c[:,0,:],axis=0)).astype(int)
def dist(pt1, pt2):
    return np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)

try:
    condition = sys.argv[1]
    mode = int(sys.argv[2])
except IndexError:
    print "Usage: python track_mouse3.py condition [0:control 1:exp] <optional: data location>\nExample:\npython track_mouse3.py Black6_5 0 /Volumes/Flashdrive/"
    sys.exit(0)
try:
    datadir = sys.argv[3]
except IndexError:
    datadir = '.'

CONTROL = 0
TEST = 1

startdir = os.getcwd()
os.chdir(datadir)
if mode==CONTROL:
    baseline_dir = condition+'_BG'
    trialdir = condition+'_BS'
elif mode==TEST:
    baseline_dir = condition+'_test_BG'
    trialdir = condition+'_test'

try:
    baseline = np.load(baseline_dir+'/%s_baseline.npy'%baseline_dir)
except:
    blmov = cv2.VideoCapture(os.path.join(baseline_dir, baseline_dir+'-cam0.avi'))
    valid, baseline = get_frame(blmov, n=-1)
    blmov.release()
    np.save(baseline_dir+'/%s_baseline'%baseline_dir, baseline)
    
    blmov = cv2.VideoCapture(os.path.join(baseline_dir, baseline_dir+'-cam0.avi'))
    valid, baseline_orig = get_frame(blmov, n=-1, blur=False)
    blmov.release()
    np.save(baseline_dir+'/%s_baseline_orig'%baseline_dir, baseline_orig)

os.chdir(trialdir)
timefile = [i for i in os.listdir('.') if 'timestamps' in i][0]
time = json.loads(open(timefile,'r').read())[0]
vidfile = [i for i in os.listdir('.') if '.avi' in i][0]

mov = cv2.VideoCapture(vidfile)
#for i in range(1500):
#    mov.read() #for testing
height, width = np.shape(baseline)
track = np.zeros(np.shape(baseline))
valid,first = get_frame(mov)
try:
    pts = np.load('%s_pts.npz'%trialdir)
    pts_l = pts['pts_l']
    pts_r = pts['pts_r']
    pts_mouse = pts['pts_mouse']
except:
    pl.imshow(first, cmap=mpl_cm.Greys_r)
    print "Select left room- around the corners in order."
    pts_l = ginput(4)
    print "Select right room- around the corners in order."
    pts_r = ginput(4)
    print "Select mouse."
    pts_mouse = ginput(4)
    pl.close()
    np.savez('%s_pts'%trialdir, pts_l=pts_l, pts_r=pts_r, pts_mouse=pts_mouse)
path_l, path_r = [mpl_path.Path(pts) for pts in [pts_l,pts_r]]
last_center = np.round(np.mean(pts_mouse, axis=0)).astype(int)
allpoints = np.append(pts_l,pts_r,axis=0)
left_border = np.min([i[0] for i in allpoints])
right_border = np.max([i[0] for i in allpoints])
top_border = np.min([i[1] for i in allpoints])
bottom_border = np.max([i[1] for i in allpoints])
allpath = mpl_path.Path([[left_border,top_border],[right_border,top_border],[right_border,bottom_border],[left_border,bottom_border]])

rooms_mask = np.zeros(np.shape(baseline))
for row in range(height):
    for col in range(width):
        pt = [col,row]
        rooms_mask[row][col] = allpath.contains_point(pt)

idx = 0
widgets=[' Iterating through images...', Percentage(), Bar()]
pbar = ProgressBar(widgets=widgets, maxval=len(time)/RESAMPLE).start()

while True:
    valid,frame = get_frame(mov,skip=RESAMPLE-1)
    if not valid:
        break
    diff = cv2.absdiff(frame,baseline)
    _, diff = cv2.threshold(diff, diff_thresh, 1, cv2.THRESH_BINARY)
    diff = diff*rooms_mask
    edges = cv2.Canny(diff.astype(np.uint8), cth1, cth2)
    contours, hier = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    possible = contours
    #possible = [b for b in contours if contour_center(b)[0] > left_border and contour_center(b)[0] < right_border and contour_center(b)[1]>top_border and contour_center(b)[1]<bottom_border]
    possible = [c for c in possible if dist(contour_center(c),last_center)<translation_thresh]
    if len(possible) == 0:
        center = last_center
        skipped += 1
        if path_l.contains_point(center):
            left_assumed+=1
        if path_r.contains_point(center):
            right_assumed+=1
    else:
        chosen = possible[np.argmax([cv2.contourArea(c) for c in possible])]   
        center = contour_center(chosen)
        centers.append(center)
        track[center[1],center[0]] += 1
        if path_l.contains_point(center):
            left+=1
        if path_r.contains_point(center):
            right+=1
    
    #display
    if SHOW:
        showimg = np.copy(frame).astype(np.uint8)
        if path_l.contains_point(center):
            color = (0,0,0)
        elif path_r.contains_point(center):
            color = (255,255,255)
        else:
            color = (120,120,120)
        cv2.circle(showimg, tuple(center), radius=10, thickness=5, color=color)
        cv2.imshow('img',showimg)
        cv2.imshow('diff', diff)
        cv2.waitKey(1)
    #/display

    idx += 1
    last_center = center
    pbar.update(idx)
pbar.finish()

print "Skipped %i frames."%skipped
results = {}
results['n_frames'] = idx
results['n_skipped'] = skipped
results['frames_left'] = left
results['frames_right'] = right
results['frames_left-asssumed'] = left_assumed
results['frames_right-assumed'] = right_assumed
results['resample'] = RESAMPLE
results['centers'] = [list(c) for c in centers]
results['params'] = [thresh, duration, seg_thresh, translation_thresh, cth1, cth2, diff_thresh, kernel]
results['params_key'] = ['thresh','duration','seg_thresh','translation_thresh','cth1','cth2','diff_thresh','kernel']
resfile = open('%s_tracking_results.json'%trialdir,'w')
resfile.write("%s"%json.dumps(results))
resfile.close()
np.save('%s_tracking'%trialdir,track)

os.chdir(startdir)
mov.release()
cv2.destroyAllWindows()

