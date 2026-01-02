#!/home/anik/anaconda3/envs/main/bin/python

'''_________________________________________________________________________________________________
Function to correct the time for the drop frames

Anik Mandal | Oct  29, 2025
_________________________________________________________________________________________________'''


import numpy as np
import random as rd

drop_rate = np.average([28.06, 29.08, 30.56])

def corrected_time(end_frame, start_frame=0, fps = 100, n_samples = 1000):
    ts = list(range(start_frame, int(end_frame * 100/(100-drop_rate)), 1))
    t_samples = np.array([sorted(rd.sample(ts, end_frame)) for i in range(n_samples)])
    
    t_med = np.median(t_samples, axis=0)/fps
    t_std  = np.std(t_samples, axis=0)/fps
    return t_med, t_std

N_frames = {
    '60_1' : 6354,
    '60_2' : 5408,
    '60_3' : 5850,
    '60_4' : 8476,
    '60_5' : 5328,
    '40_1' : 5271,
    '40_2' : 5172,
    '40_3' : 5375,
    '40_4' : 5274,
    '40_5' : 5765,
    '20_1' : 2847,
    '20_2' : 2777,
    '20_3' : 3491,
    '20_4' : 4378,
    '20_5' : 3270,
    '10_1' : 2658,
    '10_2' : 2833,
    '10_3' : 2709,
    '10_4' : 2933,
    '10_5' : 3890
}

start_frames = {
    '60_1' : 56,
    '60_2' : 95,
    '60_3' : 125,
    '60_4' : 105,
    '60_5' : 114,
    '40_1' : 409,
    '40_2' : 74,
    '40_3' : 59,
    '40_4' : 102,
    '40_5' : 73,
    '20_1' : 134,
    '20_2' : 157,
    '20_3' : 131,
    '20_4' : 168,
    '20_5' : 123,
    '10_1' : 206,
    '10_2' : 211,
    '10_3' : 190,
    '10_4' : 226,
    '10_5' : 190
} 