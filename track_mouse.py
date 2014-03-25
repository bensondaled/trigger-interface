import numpy as np
import sys
import pylab as pl
pl.ion()
import cv2
cv = cv2.cv

thresh = 10
duration = 1
delta_max = 100
delta_min = 10

try:
    filename = sys.argv[1]
except IndexError:
    print "Supply movie data file as argument."
    sys.exit(0)

#imgs = np.load(filename)['data']
#imgs = [np.squeeze(im.astype(np.uint8)) for im in np.split(imgs,np.shape(imgs)[0],axis=0)]

centers = []

height, width = imgs[0].shape[:2]
motion_history = np.zeros((height, width), np.float32)
for idx,img in zip(range(1,len(imgs)), imgs[1:]):
    diff = cv2.absdiff(img, imgs[idx-1])
    _, motion_mask = cv2.threshold(diff, thresh, 1, cv2.THRESH_BINARY)
    cv2.updateMotionHistory(motion_mask, motion_history, idx, duration)
    gradient_mask, gradient_orientation = cv2.calcMotionGradient(motion_history, delta_max, delta_min)
    seg_mask, seg_bounds = cv2.segmentMotion(motion_history, idx, delta_max)
    if len(seg_bounds):
        largest_idx = np.argmax([b[2]*b[3] for b in seg_bounds])
        largest = seg_bounds[largest_idx]
        pl.clf()
        pl.imshow(img)
        pl.gca().add_patch(pl.Rectangle(largest[:2],largest[-2],largest[-1]))
        print largest
        raw_input()
        centers.append((int(largest[0]+round(0.5*largest[2])), int(largest[1]+round(0.5*largest[3]))))
    else:
        #print "skipped a frame because no difference"
        pass

mean_img = np.mean(np.dstack(imgs),axis=2)
[cv2.circle(mean_img, c, radius=3, color=(20,20,20)) for c in centers]
cv2.imwrite('test.png',mean_img)
