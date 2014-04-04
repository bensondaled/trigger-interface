from mouse_tracker import Analysis
from cv2 import resize
from scipy.ndimage.filters import gaussian_filter as gf
import numpy as np
import pylab as pl
from matplotlib import cm as mpl_cm

mice = ['Black6_9', 'Black6_10', 'Black6_11', 'Black6_7', 'Black6_8',  'Black6_13', 'Black6_14', 'Black6_15']

sigma = 1.5

results = []

for mode in [0,1]:
    heats = []
    pics = []

    for mouse in mice:
        a = Analysis(mouse, mode, data_directory='/Volumes/COMPATIBLE/April4-data')
        pic,heat,nframes,centers = a.make_fig1()
        heats.append(heat/nframes)
        pics.append(pic)

    minheight, minwidth = min([np.shape(h)[0] for h in heats]), min([np.shape(h)[1] for h in heats])
    for idx,h in enumerate(heats):
        heats[idx] = resize(h, (minwidth,minheight))
        pics[idx] = resize(pics[idx], (minwidth,minheight))

    heat = np.dstack(heats)
    img = np.dstack(pics)
    img = np.mean(img, axis=2)
    avg = np.mean(heat, axis=2)
    heat = gf(avg,sigma)
    heat = heat/np.max(heat)
    results.append([img,heat])

pl.figure()
pl.imshow(results[0][0], cmap=mpl_cm.Greys_r)
pl.imshow(np.ma.masked_where(results[0][1]<np.percentile(results[0][1],50),results[0][1]), cmap=mpl_cm.jet)
#pl.imshow(results[0][1])
pl.figure()
pl.imshow(results[1][0], cmap=mpl_cm.Greys_r)
pl.imshow(np.ma.masked_where(results[1][1]<np.percentile(results[1][1],70),results[1][1]), cmap=mpl_cm.jet)
#pl.imshow(results[1][1])
