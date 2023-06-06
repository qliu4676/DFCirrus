import os
import shutil
import pickle

import numpy as np
from astropy.io import fits

### fits helper function ###

def get_frame_array(path_or_array):
    """ Get array value for frame path or array """
    if isinstance(path_or_array, str):
        data = fits.getdata(path_or_array)
    elif isinstance(path_or_array, np.ndarray):
        data = path_or_array.copy()
    else:
        raise ValueError("Input is neither path nor numpt array!")
    return data

def get_header_val(frame, key):
    """ Get header value for frame path """
    header = fits.getheader(frame)
    val = header.get(key)
    if val is not None:
        return val
    else:
        return np.nan
        

### File manipulation helper ###

def redirect_path(path, dst):
    """ Return a new path name under a designated directory """
    basename = os.path.basename(path)
    return os.path.join(dst, basename)

def get_segm_name(frame, segm_path):
    """ Get segmentation file name of the light file. """
    return redirect_path(frame.replace('_light', '_segm'), segm_path)

def move_dir(src_path, dst_path):
    """ Move the target directory under a new directory. """
    if os.path.exists(dst_path): 
        print(f'{dst_path} exists. Remove existed files.')
        shutil.rmtree(dst_path)

    shutil.copytree(src_path, dst_path)
    shutil.rmtree(src_path)  

def save_sky_stats(filename, sky_stats, landmarks_world):
    """ Save the sky measurement and landmarks to a pickle file. """
    with open(filename, 'wb') as f:
        arr = {'sky_stats':sky_stats, 'landmarks_world':landmarks_world}
        pickle.dump(arr, f, pickle.HIGHEST_PROTOCOL)
        print(f'Saved to {filename}')

def load_sky_stats(filename):
    """ Load the sky measurement and landmarks from a pickle file. """
    print(f'Loading local sky measurements {filename}')
    with open(filename, 'rb') as f:
        arr = pickle.load(f)
        try:
            landmarks_world = arr['landmarks_world']
            sky_stats = arr['sky_stats']
        except KeyError:
            print(f'{filename} does not have proper keys.')

    return sky_stats, landmarks_world
