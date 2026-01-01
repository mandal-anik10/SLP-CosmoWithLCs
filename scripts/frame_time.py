#!/home/anik/anaconda3/envs/main/bin/python

'''_________________________________________________________________________________________________
Function to correct the time for the drop frames

Anik Mandal | Oct  29, 2025
_________________________________________________________________________________________________'''


import numpy as np
import random as rd

drop_rate = np.average([28.06, 29.08, 30.56])

def time_(end_frame, start_frame=0, fps = 100, n_samples = 10000):
    ts = list(range(start_frame, int(end_frame * 100/(100-drop_rate)), 1))
    t_samples = np.array([sorted(rd.sample(ts, end_frame)) for i in range(n_samples)])
    
    t_mean = np.median(t_samples, axis=0)/fps
    t_std  = np.std(t_samples, axis=0)/fps
    return t_mean, t_std