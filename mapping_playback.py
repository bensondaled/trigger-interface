import numpy as np
import pylab as pl
import cv2
import sys
import os
pjoin = os.path.join
TRIG_DELAY = 1.4

class FileHandler(object):
    CAM = 0
    TIME = 1
    def __init__(self, data_dir, name):
        self.data_dir = data_dir
        self.name = name
        self.n_trials = self.get_n_trials()
        self.n_cams = self.get_n_cams()
    def get_n_trials(self):
        base = pjoin(self.data_dir,self.name,self.name)
        i = 1
        while True:
            name = pjoin(base+'_%02d_cam1.avi'%i)
            if os.path.exists(name):
                pass
            else:
                break
            i+=1
        return i-1
    def get_n_cams(self):
        if self.n_trials == 0:
            return 0
        base = pjoin(self.data_dir,self.name,self.name)
        i = 1
        while True:
            name = pjoin(base+'_01_cam%i.avi'%i)
            if os.path.exists(name):
                pass
            else:
                break
            i+=1
        return i-1
    def get_path(self, trialn, mode=0, camn=1):
        base = pjoin(self.data_dir,self.name,self.name)
        if mode == self.CAM:
            suffix='cam%i.avi'%camn
        elif mode == self.TIME:
            suffix='timestamps.npz'
        name = pjoin(base+'_%02d_%s'%(trialn,suffix))
        return name

def plot(name, n):
    pl.ioff()
    fh = FileHandler(pjoin('.','data'), name)
    if fh.n_trials < 1:
        return
    n = min(n,fh.n_trials)
    #allmeans = []
    #allts = []
    for i in xrange(n):
        vc = cv2.VideoCapture(fh.get_path(fh.n_trials-i,fh.CAM,1))
        t = np.load(fh.get_path(fh.n_trials-i,fh.TIME))
        ts = t['time1']
        ts = [i[1] if type(i) in [list,np.ndarray] else i for i in ts]
        trigt = t['trigger'][0] - ts[0] + TRIG_DELAY
        ts = ts-ts[0]
        meanf = []
        for i in xrange(len(ts)):
            valid,fr = vc.read()
            meanf.append(np.mean(fr))
        #allmeans.append(meanf)
        #allts.append(ts)
        pl.plot(ts,meanf)
        pl.plot((trigt,trigt),(0,max(meanf)),'k--')
    pl.gca().set_xlim(left=-0.1)
    pl.show()
    
        
        
def play(name):
    fh = FileHandler(pjoin('.','data'), name)
    vcs = []
    if fh.n_trials < 1:
        return
    for cam in xrange(1,fh.n_cams+1):
        vc = cv2.VideoCapture(fh.get_path(fh.n_trials,fh.CAM,cam))
        vcs.append(vc)
    t = np.load(fh.get_path(fh.n_trials,fh.TIME))
    ts = [t['time%i'%i] for i in xrange(1,fh.n_cams+1)]
    t0s = [i[0][1] if type(i[0]) in [list,np.ndarray] else i[0] for i in ts]

    width = 430
    for i in xrange(1,fh.n_cams+1):
        cv2.namedWindow(str(i))
        cv2.moveWindow(str(i), 5+width*(i-1)+(i-1)*20,5)
    Ts = 20
    while True:
        anytrue = False
        mint = np.min([st[1] if len(st)==3 else st[0] for st in [t[0] for t in ts]])
        for vci,vc,t in zip(xrange(len(vcs)),vcs,ts):
            qtime = t[0]
            if len(qtime) == 3:    
                qtime=qtime[1]
            elif len(qtime) == 2:
                qtime=qtime[0]
            if qtime>mint:
                continue
            valid,frame = vc.read()
            if valid:
                tstamp = t[0]
                if len(t)>1:
                    ts[vci]=ts[vci][1:]
                if type(tstamp) in [list,np.ndarray]:
                    tstamp = tstamp[1]
                rsf = frame.shape[1]/float(width)
                newshape = np.round(np.array(frame.shape[:-1])[::-1]/rsf).astype(int)
                frame = cv2.resize(frame, tuple(newshape))
                cv2.putText(frame,'%0.4f'%(tstamp-t0s[vci]),(5,30),0,1,(160,100,80))
                cv2.imshow(str(vci+1), frame)
                anytrue = True
        k=cv2.waitKey(Ts)
        if k == ord('f'):
            Ts = max(1,Ts-10)
        elif k==ord('s'):
            Ts = Ts+10
        elif k==ord('q'):
            break
        if not anytrue:
            break
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    if len(sys.argv)>2:
        name = sys.argv[1]
        mode = sys.argv[2]
        if mode=='play':
            play(name)
        elif mode=='plot':
            if len(sys.argv)>3:
                n=int(sys.argv[3])
            else:
                n = 1
            plot(name,n)
