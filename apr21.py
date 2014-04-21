from tsc_exp_analyze import Analysis
import cv2
from scipy.ndimage.filters import gaussian_filter as gf

sigma = 1.5
results = []
datadir = '.'
mice = ['Black6_5']

BASELINE = 0
TEST = 1

IMG = 0
HEAT = 1

for mode in [BASELINE, TEST]:
    heats = []
    pics = []

    for mouse in mice:
        a = Analysis(mouse, mode, data_directory=datadir)
        bg,heat = a.get_bg_tr_cropped()
        heats.append(heat)
        pics.append(bg)

    minheight, minwidth = min([np.shape(h)[0] for h in heats]), min([np.shape(h)[1] for h in heats])
    for idx,h in enumerate(heats):
        heats[idx] = cv2.resize(h, (minwidth,minheight))
        pics[idx] = cv2.resize(pics[idx], (minwidth,minheight))

    heat = np.dstack(heats)
    img = np.dstack(pics)
    img = np.mean(img, axis=2)
    avg = np.mean(heat, axis=2)
    heat = gf(avg,sigma)
    heat = heat/np.max(heat)
    results.append([img,heat])

    pl.figure()
    pl.imshow(results[BASELINE][IMG], cmap=mpl_cm.Greys_r)
    pl.imshow(np.ma.masked_where(results[BASELINE][HEAT]<np.percentile(results[BASELINE][HEAT],50),results[BASELINE][HEAT]), cmap=mpl_cm.jet)
    #pl.imshow(results[0][1])
    pl.figure()
    pl.imshow(results[TEST][IMG], cmap=mpl_cm.Greys_r)
    pl.imshow(np.ma.masked_where(results[TEST][HEAT]<np.percentile(results[TEST][HEAT],70),results[TEST][HEAT]), cmap=mpl_cm.jet)
    #pl.imshow(results[1][1])

