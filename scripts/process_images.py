#!/home/anik/anaconda3/envs/main/bin/python

'''_________________________________________________________________________________________________
Script to multiprocess all the experimental images and to find the string length 

Anik Mandal | Oct  29, 2025
_________________________________________________________________________________________________'''


import numpy as np
from scipy.stats import norm
from tqdm import tqdm
import multiprocessing
import json

from skimage import io
from skimage.filters import meijering, apply_hysteresis_threshold, gaussian, median
from skimage.morphology import binary_closing, disk, remove_small_objects, skeletonize
from skan import Skeleton, summarize

import warnings
warnings.filterwarnings('ignore')


file_path = '../exp-data/20251201/28_ramp/1000_1'
output_file = '1000_1_28_processed'
num_imgs = 1000

# Calibration scale
calib_scale = 0.068e-3    # mm/px
effectiv_cell_area = 8e-5    # m^2
C = 16.23e-12    # f
cell_thickness = (effectiv_cell_area * 8.85e-12 / C) * 1e3    # mm


def framework(image, median_footprint=3, ridge_sigma=range(1, 4, 1), hysteresis_th=[0.08, 0.2], closing_footprint=3, obj_th=64, skleton_img=False):
    '''
    Main image analysis framework function which processes a given image.
    '''
    # Turning image to float format
    if image.dtype!= 'float':
        image = image.astype(float) / 255.0

    # Step-1 : Applying noise filters:
    denoised_image = median(image, footprint=np.ones(( median_footprint,  median_footprint)))

    # Step-2 : Detecting defect strings as ridge-like structure
    ridge_map = meijering(denoised_image, sigmas = ridge_sigma, black_ridges = True)
    
    # Step-3 : Segmenting with hysteresis
    binary_ridge_map = apply_hysteresis_threshold( ridge_map, low = hysteresis_th[0], high = hysteresis_th[1])

    # Step-4 : Refining and cleaning 
    closed_map = binary_closing(binary_ridge_map, footprint = disk((closing_footprint)))
    cleaned_map = remove_small_objects(closed_map, min_size = obj_th)

    # Step-5 : Skeletonizing and measuring total length
    skleton = skeletonize(cleaned_map)
    skeleton_obj = Skeleton(skleton)
    summary_df  = summarize(skeleton_obj)
    total_length = summary_df['branch-distance'].sum()
    
    string_density = total_length  /  (image.size * calib_scale * cell_thickness)
    
    if skleton_img == True:
        return string_density, skleton
    else:
        return string_density
          

def perform_processing(idx):
    t_frame = idx/100
    fig_name = file_path + "/{:.2f}s.tif".format(t_frame)
    try: 
        img = io.imread(fig_name, as_gray=True)
        out = framework(img, hysteresis_th=[0.15, 0.25])
        return [t_frame, out]
    except:
        return


## processing::====================================================================================
if __name__ == '__main__':
    cores = multiprocessing.cpu_count()   # Define number of cores for multiprocessing.
    
    with multiprocessing.Pool(cores) as pool:
        outs = [result for result in tqdm(pool.imap(perform_processing, range(0, num_imgs, 1)), total=num_imgs) 
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

if __name__ == '__main__':
    output_file = '../results/' + output_file
    
    with open('%s.json'%output_file, 'w') as file:
        json.dump(outs, file, indent=4, cls=NpEncoder)
    
#============================================|THE END|=============================================

