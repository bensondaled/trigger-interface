#get times for multple mice

from mouse_tracker import Analysis
import numpy as np
import pylab as pa
from csv import DictWriter as DW

mice = ['Black6_9', 'Black6_10', 'Black6_11', 'Black6_7', 'Black6_8',  'Black6_13', 'Black6_14', 'Black6_15']

results = DW(open('/Users/Benson/Desktop/times.csv','w'), fieldnames=['Mouse','trial-type','left\%','right\%','middle\%','left', "right",'middle','total'])
results.writeheader()

for mouse in mice:
    for mode in [0,1]:
        a = Analysis(mouse, mode, data_directory='/Volumes/COMPATIBLE/April4-data')
        time, nframes, left, right, middle, resample, c, ca =  a.get_times()
        time = np.array(time)
        Ts = np.mean(time[1:]-time[:-1])
        newTs = float(Ts * resample)
        dic = {}
        left = float(left)
        right = float(right)
        middle=float(middle)
        total = (left+right+middle)*newTs
        dic['Mouse'] = mouse
        dic['trial-type'] = ['baseline','test'][mode]
        dic['left'] = left*newTs
        dic['right'] = right*newTs
        dic['middle'] = middle*newTs
        dic['total'] = total
        dic['left\%'] = left*newTs/total*100.
        dic['right\%'] = right*newTs/total*100.
        dic['middle\%'] = middle*newTs/total*100.
        results.writerow(dic)

