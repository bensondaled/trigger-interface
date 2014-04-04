from scipy.io import savemat
from tkFileDialog import askopenfilename, askdirectory
import numpy as np
from matplotlib import path as mpl_path
import matplotlib.cm as mpl_cm
from progressbar import *
import sys
import os
import json
from pylab import ion, title, close
from pylab import imshow as plimshow
from pylab import ginput as plginput
from pylab import savefig as plsavefig
from pylab import close as plclose
ion()
from cv2 import getRotationMatrix2D, warpAffine, namedWindow, VideoCapture, destroyAllWindows, cvtColor, GaussianBlur, VideoWriter, absdiff, threshold, THRESH_BINARY, Canny, findContours,  RETR_EXTERNAL, CHAIN_APPROX_TC89_L1, contourArea, circle, waitKey
from cv2 import imshow as cv2imshow
from cv2.cv import CV_RGB2GRAY, CV_FOURCC, CV_GRAY2RGB
import ttk
import Tkinter as tk

CONTROL = 0
TEST = 1
BACKGROUND = 0
TRIAL = 1
DIR = 0
NAME = 1

def ginput(n):
    pts = plginput(n, timeout=-1)
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
        
        try:
            self.items[CONTROL][BACKGROUND][NAME] = mouse+'_BG'
            self.items[CONTROL][BACKGROUND][DIR] = os.path.join(data_dir, self.items[CONTROL][BACKGROUND][NAME])
            self.items[CONTROL][TRIAL][NAME] = mouse+'_BS'
            self.items[CONTROL][TRIAL][DIR] = os.path.join(data_dir, self.items[CONTROL][TRIAL][NAME])
            self.items[TEST][BACKGROUND][NAME] = mouse+'_test_BG'
            self.items[TEST][BACKGROUND][DIR] = os.path.join(data_dir, self.items[TEST][BACKGROUND][NAME])
            self.items[TEST][TRIAL][NAME] = mouse+'_test'
            self.items[TEST][TRIAL][DIR] = os.path.join(data_dir, self.items[TEST][TRIAL][NAME])
        except:
            raise Exception('Files were not named and/or located properly.')
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

        tracking = np.load(os.path.join(self.trial_dir,'%s_tracking.npz'%self.trial_name))
        track = tracking['heat']
        nframes = tracking['n_frames']
        skipped = tracking['skipped']
        frames = nframes-skipped
        centers = tracking['centers']
        
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

        #leftmost = pts_l[np.argsort([p[0] for p in pts_l])][:2]
        #rightmost = pts_r[np.argsort([p[0] for p in pts_r])][-2:]
        #topleft = leftmost[np.argmin([p[1] for p in leftmost])]
        #topright = rightmost[np.argmin([p[1] for p in rightmost])]
        #theta = np.degrees(np.arctan2(topright[1]-topleft[1], topright[0]-topleft[0]))

        #rot_mat = getRotationMatrix2D(center=(0,0), angle=theta, scale=1)
        #bl_rotated = warpAffine(background_crop, M=rot_mat, dsize=np.shape(background_crop)[::-1])
        #tr_rotated = warpAffine(track_crop, M=rot_mat, dsize=np.shape(background_crop)[::-1])
        #plimshow(bl_rotated, cmap=mpl_cm.Greys_r)
        #plimshow(np.ma.masked_where(tr_rotated==0.,tr_rotated), cmap=mpl_cm.jet, interpolation=None)
        #plsavefig(self.trial_dir+'/%s_summary-rotation.png'%self.trial_name)
        plimshow(background_crop, cmap=mpl_cm.Greys_r)
        plimshow(np.ma.masked_where(track_crop==0., track_crop), cmap=mpl_cm.jet)
        plsavefig(self.trial_dir+'/%s_summary.png'%self.trial_name)
        plclose()
        return (background_crop,track_crop,frames,centers)

class MouseTracker(object):
    def __init__(self, mouse, mode,  data_directory='.', diff_thresh=100, resample=8, translation_max=100, smoothing_kernel=19, consecutive_skip_threshold=2):
        self.mouse = mouse
        self.data_dir = data_directory
        
        # Parameters (you may vary)
        self.diff_thresh = diff_thresh
        self.resample = resample
        self.translation_max = translation_max
        self.kernel = smoothing_kernel
        self.consecutive_skip_threshold = (37./self.resample) * consecutive_skip_threshold

        # Parameters (you should not vary)
        self.duration = 1
        self.cth1 = 0
        self.cth2 = 0
        plat = sys.platform
        if 'darwin' in plat:
            self.fourcc = CV_FOURCC('m','p','4','v') 
        elif plat[:3] == 'win':
            self.fourcc = 1
        else:
            self.fourcc = -1
        
        
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
        self.mov = VideoCapture(vidfile)
        
        self.results = {}
        self.results['centers'] = []
        self.results['centers_all'] = []
        self.results['left'] = 0
        self.results['right'] = 0
        self.results['left_assumed'] = 0
        self.results['right_assumed'] = 0
        self.results['skipped'] = 0
        self.results['heat'] = np.zeros(np.shape(self.background))
        self.results['n_frames'] = 0
        self.results['params'] = [self.diff_thresh, self.kernel, self.translation_max, self.resample]
        self.results['params_key'] = ['diff_thresh','kernel','translation_max','resample']

        self.path_l, self.path_r, self.rooms_mask, self.paths_ignore, self.last_center = self.get_pt_selections()
    def end(self):
        np.savez(os.path.join(self.trial_dir,'%s_tracking'%self.trial_name), **self.results)
        savemat(os.path.join(self.trial_dir,'%s_tracking'%self.trial_name), self.results)
        
        self.mov.release()
        destroyAllWindows()
    def get_pt_selections(self):
        valid,first = self.get_frame(self.mov, blur=False, n=30)
        try:
            pts = np.load(os.path.join(self.trial_dir, '%s_selections.npz'%self.trial_name))
            pts_l = pts['pts_l']
            pts_r = pts['pts_r']
            pts_mouse = pts['pts_mouse']
            regions_ignore = pts['regions_ignore']
        except:
            plimshow(first, cmap=mpl_cm.Greys_r)
            title("Select left room- around the corners in order.")
            pts_l = ginput(4)
            title("Select right room- around the corners in order.")
            pts_r = ginput(4)
            title("Select mouse.")
            pts_mouse = ginput(4)
            
            valid,frame = self.get_frame(self.mov)
            diff = absdiff(frame,self.background)
            _, diff = threshold(diff, self.diff_thresh, 1, THRESH_BINARY)
            plimshow(np.ma.masked_where(diff==0, diff), cmap=mpl_cm.jet)
            regions_ignore = []
            pts_ignore = [1]
            title("Select cups and any colored regions in the rooms that are not the mouse. (8pts per)")
            while True:
                pts_ignore = ginput(8)
                if not pts_ignore.any():
                    break
                regions_ignore.append(pts_ignore)
            regions_ignore = np.array(regions_ignore)
            close()
            np.savez(os.path.join(self.trial_dir, '%s_selections'%self.trial_name), pts_l=pts_l, pts_r=pts_r, pts_mouse=pts_mouse, regions_ignore=regions_ignore)
        path_l, path_r = [mpl_path.Path(pts) for pts in [pts_l,pts_r]]
        last_center = np.round(np.mean(pts_mouse, axis=0)).astype(int)
        allpoints = np.append(pts_l,pts_r,axis=0)
        left_border = np.min([i[0] for i in allpoints])
        right_border = np.max([i[0] for i in allpoints])
        top_border = np.min([i[1] for i in allpoints])
        bottom_border = np.max([i[1] for i in allpoints])
        allpath = mpl_path.Path([[left_border,top_border],[right_border,top_border],[right_border,bottom_border],[left_border,bottom_border]])

        paths_ignore = [mpl_path.Path(pts) for pts in regions_ignore]

        rooms_mask = np.zeros(np.shape(self.background))
        for row in range(self.height):
            for col in range(self.width):
                pt = [col,row]
                rooms_mask[row][col] = allpath.contains_point(pt)

        return (path_l, path_r, rooms_mask, paths_ignore, last_center)
    def load_background(self):
        try:
            bg = np.load(self.background_dir+'/%s_background.npz'%self.background_name)
            background = bg['computations']
            background_image = bg['image']
        except:
            #print "Acquiring background information..."
            blmov = VideoCapture(os.path.join(self.background_dir, self.background_name+'-cam0.avi'))
            valid, background = self.get_frame(blmov, n=-1, blur=True)
            blmov.release()
            
            blmov = VideoCapture(os.path.join(self.background_dir, self.background_name+'-cam0.avi'))
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
            frame = cvtColor(frame, CV_RGB2GRAY)
            if blur:
                frame = GaussianBlur(frame, (self.kernel,self.kernel), 0)
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
    def run(self, show=False, save=False, tk_var_frame=None):
        if show:
            namedWindow('Movie')
            namedWindow('Tracking')
        if save:
            bgs = np.shape(self.background)
            fsize = (bgs[0], bgs[1]*2)
            writer = VideoWriter()
            writer.open(os.path.join(self.trial_dir,'%s_tracking_movie'%self.trial_name),self.fourcc,37.,frameSize=(fsize[1],fsize[0]),isColor=True)

        self.results['n_frames'] = 0
        consecutive_skips = 0
        while True:
            valid,frame = self.get_frame(self.mov,skip=self.resample-1)
            if not valid:
                break
            diff = absdiff(frame,self.background)
            _, diff = threshold(diff, self.diff_thresh, 1, THRESH_BINARY)
            diff = diff*self.rooms_mask
            edges = Canny(diff.astype(np.uint8), self.cth1, self.cth2)
            contours, hier = findContours(edges, RETR_EXTERNAL, CHAIN_APPROX_TC89_L1)
            contours = [c for c in contours if not any([pa.contains_point(contour_center(c)) for pa in self.paths_ignore])]
            if consecutive_skips>self.consecutive_skip_threshold:
                consecutive_skips=0
                possible = contours
            else:
                possible = [c for c in contours if dist(contour_center(c),self.last_center)<self.translation_max]
            
            if len(possible) == 0:
                center = self.last_center
                self.results['skipped'] += 1
                consecutive_skips+=1
                if self.path_l.contains_point(center):
                    self.results['left_assumed']+=1
                if self.path_r.contains_point(center):
                    self.results['right_assumed']+=1
            else:
                chosen = possible[np.argmax([contourArea(c) for c in possible])]   
                center = contour_center(chosen)
                self.results['centers'].append(center)
                self.results['heat'][center[1],center[0]] += 1
                if self.path_l.contains_point(center):
                    self.results['left']+=1
                if self.path_r.contains_point(center):
                    self.results['right']+=1
            self.results['centers_all'].append(center) 
            #display
            if show or save:
                showimg = np.copy(frame).astype(np.uint8)
                if self.path_l.contains_point(center):
                    color = (0,0,0)
                elif self.path_r.contains_point(center):
                    color = (255,255,255)
                else:
                    color = (120,120,120)
                circle(showimg, tuple(center), radius=10, thickness=5, color=color)
                if show:
                    cv2imshow('Movie',showimg)
                    cv2imshow('Tracking', diff)
                    waitKey(1)
            #/display
            if save:
                save_frame = np.zeros([fsize[0], fsize[1], 3],dtype=np.uint8)
                save_frame[:,:np.shape(frame)[1]] = cvtColor(showimg.astype(np.float32), CV_GRAY2RGB)
                save_frame[:,np.shape(frame)[1]:] = cvtColor((diff*255).astype(np.float32), CV_GRAY2RGB)
                writer.write(save_frame)
             
            self.results['n_frames'] += 1
            self.last_center = center
            if tk_var_frame != None:
                tk_var_frame[0].set('%i/%i'%(self.results['n_frames'], len(self.time)/float(self.resample) ))
                tk_var_frame[1].update()
            #pbar.update(self.results['n_frames'])
        #pbar.finish()
        if save:
            writer.release()
        self.end()

if __name__=='__main__':
    
    class MainFrame(object):
        def __init__(self, parent, options=[]):
            self.parent = parent
            self.frame = ttk.Frame(self.parent)
            
            self.selection = None
            self.show = tk.IntVar()
            self.save = tk.IntVar()
            self.resample = tk.StringVar()
            self.resample.set('5')
            self.diff_thresh = tk.StringVar()
            self.diff_thresh.set('80')
            self.bltest = tk.IntVar()
            
            self.initUI(options)
        def initUI(self, options):
            self.parent.title('Select Mouse')
            self.frame.pack(fill=tk.BOTH, expand=1)

            self.lb = tk.Listbox(self.frame, selectmode=tk.EXTENDED)
            for i in options:
                self.lb.insert(tk.END, i)

            self.ok = ttk.Button(self.frame, text='OK')
            self.ok.bind("<Button-1>", self.done_select)
            self.ok.bind("<Return>", self.done_select)

            self.show_widg = ttk.Checkbutton(self.frame, text='Show tracking', variable=self.show)
            self.save_widg = ttk.Checkbutton(self.frame, text='Save tracking video', variable=self.save)
            self.resample_widg = ttk.Entry(self.frame, textvariable=self.resample)
            self.diff_thresh_widg = ttk.Entry(self.frame, textvariable=self.diff_thresh)
            label1 = ttk.Label(self.frame, text='Resample:')
            label2 = ttk.Label(self.frame, text='Threshold:')
            b1 = tk.Radiobutton(self.frame, text='Baseline', variable=self.bltest, value=CONTROL)
            b2 = tk.Radiobutton(self.frame, text='Test', variable=self.bltest, value=TEST)

            self.lb.place(x=5, y=5)
            self.resample_widg.pack(side=tk.BOTTOM)
            label1.pack(side=tk.BOTTOM)
            self.diff_thresh_widg.pack(side=tk.BOTTOM)
            label2.pack(side=tk.BOTTOM)
            self.save_widg.pack(side=tk.BOTTOM)
            self.show_widg.pack(side=tk.BOTTOM)
            self.ok.pack(side=tk.BOTTOM)
            b1.pack(side=tk.RIGHT)
            b2.pack(side=tk.RIGHT)

        def done_select(self, val):
            idxs = map(int, self.lb.curselection())
            values = [self.lb.get(idx) for idx in idxs]
            
            self.selection = values

            self.frame.destroy()

            self.main()
        def main(self):
            self.parent.title('Status')
            self.frame = ttk.Frame(self.parent, takefocus=True)
            self.frame.pack(fill=tk.BOTH, expand=1)
            self.todo = tk.StringVar()
            self.status1 = tk.StringVar()
            self.status2 = tk.StringVar()
            label_todo = ttk.Label(self.frame, textvariable=self.todo)
            label1 = ttk.Label(self.frame, textvariable=self.status1)
            label2 = ttk.Label(self.frame, textvariable=self.status2)
            self.todo.set('Mice:\n'+'\n'.join(self.selection))
            label2.pack(side=tk.TOP)
            label1.pack(side=tk.TOP)
            label_todo.pack(side=tk.BOTTOM)

            mice = self.selection
            save_tracking_video = self.save.get()
            show_live_tracking = self.show.get()
            resample = int(self.resample.get())
            diff_thresh = int(self.diff_thresh.get())
            mode = self.bltest.get()
            for mouse in mice:
                self.status1.set('Now processing mouse \'%s\', %s trial.'%(mouse, {CONTROL:'baseline', TEST:'test'}[mode]))
                self.frame.update()
                try:
                    mt = MouseTracker(mouse=mouse, mode=mode, data_directory=data_dir, resample=resample, diff_thresh=diff_thresh)
                    mt.run(show=show_live_tracking, save=save_tracking_video, tk_var_frame=(self.status2, self.frame))
                    a = Analysis(mouse, mode, data_directory=data_dir)
                    a.make_fig1()
                except:
                    pass
            self.parent.destroy()
    def parse_mice_names(items):
        good = []
        for item in items:
           if '_BG' not in item:
            continue
           if 'test' in item:
            continue
           mousename = item[:item.index('_BG')]
           good.append(mousename)
        return good
    
    root = tk.Tk()
    root.geometry("400x400+200+100")
    data_dir = askdirectory(parent=root, initialdir='/', mustexist=True, title='Select directory containing data folders.')
    if not data_dir:
        sys.exit(0)
    options = [o for o in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,o))]
    options = parse_mice_names(options)

    frame = MainFrame(root, options=options)
    root.mainloop()
   

