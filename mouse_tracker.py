from scipy.io import savemat
import numpy as np
from matplotlib import path as mpl_path
import matplotlib.cm as mpl_cm
from progressbar import *
import sys
import os
import json
import pylab as pl
pl.ion()
import cv2
cv = cv2.cv

CONTROL = 0
TEST = 1
BACKGROUND = 0
TRIAL = 1
DIR = 0
NAME = 1

def ginput(n):
    pts = pl.ginput(n, timeout=-1)
    pts = np.array(pts)
    return pts
def contour_center(c):
    return np.round(np.mean(c[:,0,:],axis=0)).astype(int)
def dist(pt1, pt2):
    return np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)

class FileHandler(object):
    def __init__(self, data_dir, mouse):
        self.items = {}
        self.items[CONTROL] = {BACKGROUND:{}, TRIAL:{}}
        self.items[TEST] = {BACKGROUND:{}, TRIAL:{}}
        
        self.items[CONTROL][BACKGROUND][NAME] = mouse+'_BG'
        self.items[CONTROL][BACKGROUND][DIR] = os.path.join(data_dir, self.items[CONTROL][BACKGROUND][NAME])
        self.items[CONTROL][TRIAL][NAME] = mouse+'_BS'
        self.items[CONTROL][TRIAL][DIR] = os.path.join(data_dir, self.items[CONTROL][TRIAL][NAME])
        self.items[TEST][BACKGROUND][NAME] = mouse+'_test_BG'
        self.items[TEST][BACKGROUND][DIR] = os.path.join(data_dir, self.items[TEST][BACKGROUND][NAME])
        self.items[TEST][TRIAL][NAME] = mouse+'_test'
        self.items[TEST][TRIAL][DIR] = os.path.join(data_dir, self.items[TEST][TRIAL][NAME])
    def __getitem__(self, item):
        return self.items[item]

class Analysis(object):
    def __init__(self, mouse, mode, data_directory='.'):
        
        fh = FileHandler(data_directory, mouse)
        self.background_name = fh[mode][BACKGROUND][NAME]
        self.background_dir = fh[mode][BACKGROUND][DIR]
        self.trial_name = fh[mode][TRIAL][NAME]
        self.trial_dir = fh[mode][TRIAL][DIR]
    def make_fig1(self):
        bg = np.load(os.path.join(self.background_dir,'%s_background.npz'%self.background_name))
        background = bg['image']

        theCM = mpl_cm.get_cmap()
        theCM._init()
        alphas = np.abs(np.linspace(-1.0, 1.0, theCM.N))
        theCM._lut[:-3,-1] = alphas
        
        track = np.load(os.path.join(self.trial_dir,'%s_tracking.npz'%self.trial_name))['heat']
        
        height, width = np.shape(background)
        pts = np.load(self.trial_dir+'/%s_selections.npz'%self.trial_name)
        pts_l = pts['pts_l']
        pts_r = pts['pts_r']
        path_l, path_r = [mpl_path.Path(pts) for pts in [pts_l,pts_r]]
        allpoints = np.append(pts_l,pts_r,axis=0)
        left = np.min([i[0] for i in allpoints])
        right = np.max([i[0] for i in allpoints])
        top = np.min([i[1] for i in allpoints])
        bottom = np.max([i[1] for i in allpoints])

        background_crop = background[top:bottom, left:right]
        track_crop = track[top:bottom, left:right]

        leftmost = pts_l[np.argsort([p[0] for p in pts_l])][:2]
        rightmost = pts_r[np.argsort([p[0] for p in pts_r])][-2:]
        topleft = leftmost[np.argmin([p[1] for p in leftmost])]
        topright = rightmost[np.argmin([p[1] for p in rightmost])]
        theta = np.degrees(np.arctan2(topright[1]-topleft[1], topright[0]-topleft[0]))

        rot_mat = cv2.getRotationMatrix2D(center=(0,0), angle=theta, scale=1)
        bl_rotated = cv2.warpAffine(background_crop, M=rot_mat, dsize=np.shape(background_crop)[::-1])
        tr_rotated = cv2.warpAffine(track_crop, M=rot_mat, dsize=np.shape(background_crop)[::-1])
        pl.imshow(bl_rotated, cmap=mpl_cm.Greys_r)
        pl.imshow(np.ma.masked_where(tr_rotated==0.,tr_rotated), cmap=mpl_cm.jet, interpolation=None)

class MouseTracker(object):
    def __init__(self, mouse, mode,  data_directory='.', diff_thresh=100, resample=8, translation_max=130, smoothing_kernel=15):
        self.mouse = mouse
        self.data_dir = data_directory
        
        # Parameters (you may vary)
        self.diff_thresh = diff_thresh
        self.resample = resample
        self.translation_max = translation_max
        self.kernel = smoothing_kernel

        # Parameters (you should not vary)
        self.duration = 1
        self.cth1 = 0
        self.cth2 = 0

        cv2.namedWindow('Movie')
        cv2.namedWindow('Tracking')
        
        fh = FileHandler(self.data_dir, self.mouse)
        self.background_name = fh[mode][BACKGROUND][NAME]
        self.background_dir = fh[mode][BACKGROUND][DIR]
        self.trial_name = fh[mode][TRIAL][NAME]
        self.trial_dir = fh[mode][TRIAL][DIR]

        self.background, self.background_image = self.load_background()
        self.height, self.width = np.shape(self.background)
        
        timefile = os.path.join(self.trial_dir, self.trial_name+'-timestamps.json')
        self.time = json.loads(open(timefile,'r').read())[0]
        vidfile = os.path.join(self.trial_dir, self.trial_name+'-cam0.avi')
        self.mov = cv2.VideoCapture(vidfile)
        
        self.results = {}
        self.results['centers'] = []
        self.results['left'] = 0
        self.results['right'] = 0
        self.results['left_assumed'] = 0
        self.results['right_assumed'] = 0
        self.results['skipped'] = 0
        self.results['heat'] = np.zeros(np.shape(self.background))
        self.results['n_frames'] = 0
        self.results['params'] = [self.diff_thresh, self.kernel, self.translation_max, self.resample]
        self.results['params_key'] = ['diff_thresh','kernel','translation_max','resample']

        self.path_l, self.path_r, self.rooms_mask, self.last_center = self.get_pt_selections()
    def end(self):
        np.savez(os.path.join(self.trial_dir,'%s_tracking'%self.trial_name), **self.results)
        savemat(os.path.join(self.trial_dir,'%s_tracking'%self.trial_name), self.results)
        
        self.mov.release()
        cv2.destroyAllWindows()
    def get_pt_selections(self):
        valid,first = self.get_frame(self.mov, blur=False)
        try:
            pts = np.load(os.path.join(self.trial_dir, '%s_selections.npz'%self.trial_name))
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
            np.savez(os.path.join(self.trial_dir, '%s_selections'%self.trial_name), pts_l=pts_l, pts_r=pts_r, pts_mouse=pts_mouse)
        path_l, path_r = [mpl_path.Path(pts) for pts in [pts_l,pts_r]]
        last_center = np.round(np.mean(pts_mouse, axis=0)).astype(int)
        allpoints = np.append(pts_l,pts_r,axis=0)
        left_border = np.min([i[0] for i in allpoints])
        right_border = np.max([i[0] for i in allpoints])
        top_border = np.min([i[1] for i in allpoints])
        bottom_border = np.max([i[1] for i in allpoints])
        allpath = mpl_path.Path([[left_border,top_border],[right_border,top_border],[right_border,bottom_border],[left_border,bottom_border]])

        rooms_mask = np.zeros(np.shape(self.background))
        for row in range(self.height):
            for col in range(self.width):
                pt = [col,row]
                rooms_mask[row][col] = allpath.contains_point(pt)

        return (path_l, path_r, rooms_mask, last_center)
    def load_background(self):
        try:
            bg = np.load(self.background_dir+'/%s_background.npz'%self.background_name)
            background = bg['computations']
            background_image = bg['image']
        except:
            print "Acquiring background information..."
            blmov = cv2.VideoCapture(os.path.join(self.background_dir, self.background_name+'-cam0.avi'))
            valid, background = self.get_frame(blmov, n=-1)
            blmov.release()
            
            blmov = cv2.VideoCapture(os.path.join(self.background_dir, self.background_name+'-cam0.avi'))
            valid, background_image = self.get_frame(blmov, n=-1, blur=False)
            blmov.release()
            
            np.savez(self.background_dir+'/%s_background'%self.background_name, computations=background, image=background_image)
        return background, background_image
    def get_frame(self, mov, n=1, skip=0, blur=True):
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
                frame = cv2.GaussianBlur(frame, (self.kernel,self.kernel), 0)
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
    def run(self, show=False, save=False):

        if save:
            bgs = np.shape(self.background)
            fsize = (bgs[0], bgs[1]*2)
            writer = cv2.VideoWriter()
            writer.open(os.path.join(self.trial_dir,'%s_tracking_movie.mov'%self.trial_name),cv.CV_FOURCC('D','I','V','X'),30,frameSize=fsize,isColor=False)

        self.results['n_frames'] = 0
        widgets=[' Iterating through images...', Percentage(), Bar()]
        pbar = ProgressBar(widgets=widgets, maxval=len(self.time)/self.resample).start()

        while True:
            valid,frame = self.get_frame(self.mov,skip=self.resample-1)
            if not valid:
                break
            diff = cv2.absdiff(frame,self.background)
            _, diff = cv2.threshold(diff, self.diff_thresh, 1, cv2.THRESH_BINARY)
            diff = diff*self.rooms_mask
            edges = cv2.Canny(diff.astype(np.uint8), self.cth1, self.cth2)
            contours, hier = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
            possible = [c for c in contours if dist(contour_center(c),self.last_center)<self.translation_max]
            
            if len(possible) == 0:
                center = self.last_center
                self.results['skipped'] += 1
                if self.path_l.contains_point(center):
                    self.results['left_assumed']+=1
                if self.path_r.contains_point(center):
                    self.results['right_assumed']+=1
            else:
                chosen = possible[np.argmax([cv2.contourArea(c) for c in possible])]   
                center = contour_center(chosen)
                self.results['centers'].append(center)
                self.results['heat'][center[1],center[0]] += 1
                if self.path_l.contains_point(center):
                    self.results['left']+=1
                if self.path_r.contains_point(center):
                    self.results['right']+=1
            
            #display
            if show:
                showimg = np.copy(frame).astype(np.uint8)
                if self.path_l.contains_point(center):
                    color = (0,0,0)
                elif self.path_r.contains_point(center):
                    color = (255,255,255)
                else:
                    color = (120,120,120)
                cv2.circle(showimg, tuple(center), radius=10, thickness=5, color=color)
                cv2.imshow('Movie',showimg)
                cv2.imshow('Tracking', diff)
                cv2.waitKey(1)
            #/display
            if save:
                save_frame = np.zeros(fsize,dtype=np.uint8)
                save_frame[:,:np.shape(frame)[1]] = np.round(frame).astype(np.uint8)
                save_frame[:,np.shape(frame)[1]:] = np.round(diff).astype(np.uint8)
                writer.write(save_frame)
             
            self.results['n_frames'] += 1
            self.last_center = center
            pbar.update(self.results['n_frames'])
        pbar.finish()
        if save:
            writer.release()
        self.end()

if __name__=='__main__':
    mouse = 'Black6_5'
    mode = TEST
    data_directory = '/Volumes/BENSON32GB/'
    
    mt = MouseTracker(mouse=mouse, mode=mode, data_directory=data_directory, resample=8)
    mt.run(show=False, save=True)

    a = Analysis(mouse, mode, data_directory=data_directory)
    a.make_fig1()

