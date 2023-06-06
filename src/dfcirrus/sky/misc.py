import os
import shutil
import pickle
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import morphology

from astropy.io import fits
from astropy.stats import sigma_clip

from .io import get_frame_array, get_header_val

def symmetrize(A):
    """ Transform a upper triangle matrix into a symmetrize matrix """
    return A + A.T - np.diag(A.diagonal())

### fits helper function ###
    
def get_sky_median(frame, segm=None):
    """ Get the median sky value of the frame. """
    data = get_frame_array(frame)
    mask = np.isnan(data) 

    if segm is not None:
        mask_src = get_frame_array(segm)>0
        mask_src = morphology.binary_dilation(mask_src, iterations=3)
        mask = mask | mask_src
    
    vals = data[~mask]
    vals_clipped = sigma_clip(vals, maxiters=5).compressed()
    bkg_val = np.median(vals_clipped)
    
    return bkg_val
    
def get_sky_stats(frame, segm=None):
    """ Get the sigma-clipped median sky value and std of the frame. """
    data = get_frame_array(frame)
    mask = np.isnan(data) 

    if segm is not None:
        mask_src = get_frame_array(segm)>0
        mask_src = morphology.binary_dilation(mask_src, iterations=3)
        mask = mask | mask_src
    
    vals = data[~mask]
    vals_clipped = sigma_clip(vals, maxiters=5).compressed()
    bkg_val = np.median(vals_clipped)
    bkg_rms = np.std(vals_clipped)
    
    return bkg_val, bkg_rms

def embed_ZP(frame_paths, apass_dir=None):
    """ Calculate and embed ZP to the header for the frame list. """
    try:
        from dfreduce.tasks import calculate_zp
    except ModuleNotFoundError:
        print("calculate_zp in dfreduce not avaialble.")
    
    print('Embedding ZP...')
    for frame in tqdm(frame_paths):
        frame = str(frame)
        header = fits.getheader(frame)
        if 'REFZP' in header.keys():
            continue
        else:    
            filt = fits.getheader(frame)['FILTER']
            results = calculate_zp(frame, filt, catalogue_dir=apass_dir)
            if results is not None:
                fits.setval(frame, 'REFZP', value=results.ZP_avg)
                fits.setval(frame, 'STDZP', value=results.ZP_std)
            else:
                fits.setval(frame, 'REFZP', value=None)
                fits.setval(frame, 'STDZP', value=None)

def clip_frame_by_ZP(frame_paths, color='gray'):    
    """ Clip frames based on stddev of ZP. Return conditions, zps, mean and stddev of ZP. """
    zps = np.array([get_header_val(frame, 'REFZP') for frame in frame_paths])
    zp_stds = np.array([get_header_val(frame, 'STDZP') for frame in frame_paths])
    
    zp_med, std_zp = np.nanmedian(zps), np.nanstd(zps)

    plt.hist(zps, color=color, alpha=0.5)
    plt.axvline(zp_med, color=color, ls='-')
    plt.axvline(zp_med-3*std_zp, color=color, ls='--')
    plt.axvline(zp_med+3*std_zp, color=color, ls='--')

    cond_zp = (abs(zps-zp_med)<3*std_zp) & (zp_stds<0.2)
    zp_coadd = zps[cond_zp]
    
    return cond_zp, zp_coadd, zp_med, std_zp


### Morphology helper ###

def in_hull(p, hull):
    """
    Test if points are in hull.

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay or scipy.spatial.ConvexHull object 
    for which Delaunay triangulation will be computed.
    """
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull.points)

    return hull.find_simplex(p)>=0 
