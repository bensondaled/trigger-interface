import numpy as np
import cv2
import os
import sys

datapath = 'C:/Users/andrea/Desktop/trigger-interface/data/'
name = 'test38'
n = '05'
cam1path = os.path.join(datapath,name,name+'_%s_cam1.avi'%n)
cam2path = os.path.join(datapath,name,name+'_%s_cam2.avi'%n)
tspath = os.path.join(datapath,name,name+'_%s_timestamps.npz'%n)

ts = np.load(tspath)
t1 = ts['time1']
t2 = ts['time2']
ts1 = np.mean(t1[1:]-t1[:-1])
ts2 = np.mean(t2[1:]-t2[:-1])
trigtime = ts['trigger']
print 'trig: %0.9f'%trigtime;sys.stdout.flush()

vc1 = cv2.VideoCapture(cam1path)
vc2 = cv2.VideoCapture(cam2path)

idx=0
valid,frame = vc1.read()
while True:
    if t1[idx][1]>trigtime-4*ts1:
        cv2.putText(frame, '%0.9f'%t1[idx][1], (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 220)
        cv2.imshow('cam1', frame)
        k = cv2.waitKey(0)
        if k==ord('q'):
            break
    valid,frame = vc1.read()
    idx += 1
res1 = [t1[idx][1],t1[idx-1][1]]
print 'pgry: %0.9f (%0.9f)'%(t1[idx][1],t1[idx-1][1])
cv2.destroyWindow('cam1')

idx=0
valid,frame = vc2.read()
while True:
    if t2[idx]>trigtime-4*ts2:
        cv2.putText(frame, '%0.9f'%t2[idx], (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 220)
        cv2.imshow('cam2', frame)
        k = cv2.waitKey(0)
        if k==ord('q'):
            break
    valid,frame = vc2.read()
    idx += 1
res2 = [t2[idx], t2[idx-1]]
print 'psey: %0.9f (%0.9f)'%(t2[idx], t2[idx-1])
cv2.destroyWindow('cam2')

late = [res1,res2][np.argmax([res1[0],res2[0]])]
early = [res1,res2][np.argmin([res1[0],res2[0]])]
if (early[1]>=late[1] and early[1]<=late[0]) or (early[0]<=late[0] and early[0]>=late[1]):
    print 'ocameras agree.'
else:
    print 'cameras DISAGREE!'

print 'Delay = %0.2f - %0.2f ms'%(1000*(np.max([late[1],early[1]])-trigtime),1000*(early[0]-trigtime))