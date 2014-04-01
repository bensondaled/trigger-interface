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

theCM = mpl_cm.get_cmap()
theCM._init()
alphas = np.abs(np.linspace(-1.0, 1.0, theCM.N))
theCM._lut[:-3,-1] = alphas
cv2.namedWindow('img')

try:
    condition = sys.argv[1]
    mode = int(sys.argv[2])
except IndexError:
    print "Usage: python tracking_analysis.py condition [0:control 1:exp] <optional: data location>\nExample:\npython track_mouse3.py Black6_5 0 /Volumes/Flashdrive/"
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

baseline = np.load(baseline_dir+'/'+baseline_dir+'_baseline_orig.npy')

os.chdir(trialdir)
timefile = [i for i in os.listdir('.') if 'timestamps' in i][0]
time = np.array(json.loads(open(timefile,'r').read())[0])
Ts = np.mean(time[1:]-time[:-1])
track = np.load(trialdir+'_tracking.npy')

height, width = np.shape(baseline)
pts = np.load(trialdir+'_pts.npz')
pts_l = pts['pts_l']
pts_r = pts['pts_r']
path_l, path_r = [mpl_path.Path(pts) for pts in [pts_l,pts_r]]
allpoints = np.append(pts_l,pts_r,axis=0)
left = np.min([i[0] for i in allpoints])
right = np.max([i[0] for i in allpoints])
top = np.min([i[1] for i in allpoints])
bottom = np.max([i[1] for i in allpoints])

baseline_crop = baseline[top:bottom, left:right]
track_crop = track[top:bottom, left:right]

leftmost = pts_l[np.argsort([p[0] for p in pts_l])][:2]
rightmost = pts_r[np.argsort([p[0] for p in pts_r])][-2:]
topleft = leftmost[np.argmin([p[1] for p in leftmost])]
topright = rightmost[np.argmin([p[1] for p in rightmost])]
theta = np.degrees(np.arctan2(topright[1]-topleft[1], topright[0]-topleft[0]))

rot_mat = cv2.getRotationMatrix2D(center=(0,0), angle=theta, scale=1)
bl_rotated = cv2.warpAffine(baseline_crop, M=rot_mat, dsize=np.shape(baseline_crop)[::-1])
tr_rotated = cv2.warpAffine(track_crop, M=rot_mat, dsize=np.shape(baseline_crop)[::-1])
pl.imshow(bl_rotated, cmap=mpl_cm.Greys_r)
pl.imshow(np.ma.masked_where(tr_rotated==0.,tr_rotated), cmap=mpl_cm.jet, interpolation=None)

os.chdir(startdir)
cv2.destroyWindow('img')

