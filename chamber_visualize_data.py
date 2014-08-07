
from chamber_analyze import Analysis
import pylab as pl
from matplotlib import cm as mpl_cm
from cv2 import resize
import numpy as np
from scipy.ndimage.filters import gaussian_filter as gf

BASELINE = 0
TEST = 1
IMG = 0
HEAT = 1

def merge_mice(mice, datadir, sigma=1.5, norm=True, match_baseline_test_sizes=True):
    # Given a list of already analyzed mice, combine their data into one image and heatmap.
    # Return array: [ [[BASELINE IMG],[BASELINE HEATMAP]], [[TEST IMG],[TEST HEATMAP]] ]
    results = []

    for mode in [BASELINE, TEST]:
        heats = []
        pics = []

        for mouse in mice:
            a = Analysis(mouse, mode, data_directory=datadir)
            bg = a.get_background()['image']
            tr = a.get_tracking()
            heat = tr['heat']
            
            t = np.array(a.get_time())
            tdiff = t[1:] - t[:-1]
            assert np.std(tdiff) < 0.01
            Ts = np.mean(tdiff)
            resamp = tr['params'][np.where(tr['params_key']=='resample')]
            spf = Ts*resamp
            
            heat *= spf
            if norm:
                heat = heat/np.sum(heat)
            
            bg = a.crop(bg)
            heat = a.crop(heat)
            
            heats.append(heat)
            pics.append(bg)

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
        
    if match_baseline_test_sizes:
        heats = [results[BASELINE][HEAT], results[TEST][HEAT]]
        pics = [results[BASELINE][IMG], results[TEST][IMG]]
        minheight, minwidth = min([np.shape(h)[0] for h in heats]), min([np.shape(h)[1] for h in heats])
        for idx,h in enumerate(heats):
            heats[idx] = resize(h, (minwidth,minheight))
            pics[idx] = resize(pics[idx], (minwidth,minheight))

        results[BASELINE][HEAT], results[TEST][HEAT] = heats
        results[BASELINE][IMG], results[TEST][IMG] = pics
    return results

if __name__ == '__main__':
    # Choose the mice
    datadir = 'C:\\Users\\andreag\\Desktop\\chamber_examples' #backslashes must be doubled!
    group1 = ['mouse1']
    group2 = ['mouse3', 'mouse4']
    mice = group1

    # Parameters
    sigma = 1.5

    # Process the data
    data = merge_mice(mice=mice, datadir=datadir, sigma=sigma, norm=True)

    # Choose what to display
    bg_image = data[BASELINE][IMG]
    to_show = data[TEST][HEAT]-data[BASELINE][HEAT]
    
    # Make the data look better (option 1)
    # to_show = np.ma.masked_where(np.abs(to_show)<np.percentile(to_show, 60), to_show)
    # Make the data look better (option 2)
    to_show = np.ma.masked_where(to_show<10**-11,to_show)
    
    # Display it & save it
    pl.imshow(bg_image, cmap=mpl_cm.Greys_r)
    pl.imshow(to_show , cmap=mpl_cm.jet)
    pl.colorbar()
    pl.savefig('my_saved_image' + '.png')

