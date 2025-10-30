#!/home/anik/anaconda3/envs/main/bin/python

'''
Script to multiprocess all the experimental images and to find the string length 

Anik Mandal | Oct  29, 2025
'''


import numpy as np
from tqdm import tqdm
import multiprocessing
import json

from skimage import io
from skimage.filters import meijering, apply_hysteresis_threshold, gaussian, median
from skimage.morphology import binary_closing, disk, remove_small_objects, skeletonize
from skan import Skeleton, summarize

import warnings
warnings.filterwarnings('ignore')


file_path = '../exp-data/20251004/10V_10X/'
output_file = '10V_10X_processed'


def framework(image):
    # Turning image to float format
    if image.dtype!= 'float':
        image = image.astype(float) / 255.0

    # Step-1 : Applying noise filters:
    denoised_image = median(image, footprint=np.ones((3, 3)))

    # Step-2 : Detecting defect strings as ridge-like structure
    ridge_map = meijering(denoised_image, sigmas = range(1, 4, 1), black_ridges = True)
    
    # Step-3 : Segmenting with hysteresis
    binary_ridge_map = apply_hysteresis_threshold( ridge_map, low = 0.08, high = 0.2)

    # Step-4 : Refining and cleaning 
    closed_map = binary_closing(binary_ridge_map, footprint = disk((3)))
    cleaned_map = remove_small_objects(closed_map, min_size = 16)

    # Step-5 : Skeletonizing and measuring total length
    skleton = skeletonize(cleaned_map)
    skeleton_obj = Skeleton(skleton)
    summary_df  = summarize(skeleton_obj)
    total_length = summary_df['branch-distance'].sum()
    
    return total_length
    

def perform_processing(idx):
    t_frame = idx/100
    fig_name = file_path + "/{:.2f} s.tif".format(t_frame)
    try: 
        img = io.imread(fig_name, as_gray=True)
        out = framework(img)
        return [t_frame, out]
    except:
        return


## processing::====================================================================================
if __name__ == '__main__':
    cores = multiprocessing.cpu_count()   # Define number of cores for multiprocessing.
    
    with multiprocessing.Pool(cores) as pool:
        outs = [result for result in tqdm(pool.imap(perform_processing, range(0, 1700, 1)), total=1700) 
                if result is not None]

## Storing data::==================================================================================

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

output_file = '../results/' + output_file

with open('%s.json'%output_file, 'w') as file:
    json.dump(outs, file, indent=4, cls=NpEncoder)
    
#============================================|THE END|=============================================

