from tsc_exp_analyze import analysis1, analysis2
from tsc_exp_analyze import Analysis

BASELINE = 0
TEST = 1
IMG = 0
HEAT = 1
import pylab as pl
from matplotlib import cm as mpl_cm
from cv2 import resize
import numpy as np

sigma = 1.5
datadir = '/Volumes/LACIE SHARE/Tsc1_Social'
gr1 = ['Tsc1_103', 'Tsc1_105', 'Tsc1_223-80', 'Tsc1_240-107', 'Tsc1_248-135', 'Tsc1_260-152', 'Tsc1_261-154', 'Tsc1_276-161', 'Tsc1_279-174','Tsc1_223-86']
gr2 = ['Tsc1_221-63', 'Tsc1_248-138-2', 'Tsc1_258-144', 'Tsc1_261-151']
gr3 = ['Tsc1_104', 'Tsc1_219-145', 'Tsc1_228-95', 'Tsc1_228-99', 'Tsc1_240-109', 'Tsc1_242-115', 'Tsc1_242-116', 'Tsc1_248-132', 'Tsc1_260-149', 'Tsc1_260-153']

results = analysis1(mice=gr1, datadir=datadir, sigma=sigma, norm=True, show_perc=True)
heats = [results[BASELINE][HEAT], results[TEST][HEAT]]
pics = [results[BASELINE][IMG], results[TEST][IMG]]
minheight, minwidth = min([np.shape(h)[0] for h in heats]), min([np.shape(h)[1] for h in heats])
for idx,h in enumerate(heats):
    heats[idx] = resize(h, (minwidth,minheight))
    pics[idx] = resize(pics[idx], (minwidth,minheight))
toshow = heats[1]-heats[0]
pl.imshow(pics[0], cmap=mpl_cm.Greys_r)
toshow = np.ma.masked_where(np.abs(toshow)<0.05,toshow)
pl.imshow(toshow , cmap=mpl_cm.jet)
pl.colorbar()

#analysis2(mice=gr1+gr2+gr3, datadir=datadir)
