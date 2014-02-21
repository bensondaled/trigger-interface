import numpy as np
import matplotlib.cm as mpl_cm
import pylab as pl
import matplotlib.animation as ani

class Playback(object):
        def __init__(self, filename):
                try:
                        self.imgs = np.load(filename)
                except:
                        print "No file found for playback."
                        self.imgs = None
                
        def play(self, fps=3.):
                pl.ioff()
                if self.imgs == None:
                        return
                imgs = np.split(self.imgs,np.shape(self.imgs)[0],axis=0)
                imgs = [np.squeeze(i) for i in imgs]

                fig=pl.figure()
                ims = [[pl.imshow(i, cmap=mpl_cm.Greys_r)] for i in imgs]

                anim = ani.ArtistAnimation(fig, ims, interval=1./(fps/1000.), repeat=True)
                pl.show()
