import os
import shutil
import pickle
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.stats import mad_std

### Coadd helper function ###
def get_ccd_data(frame,
                 deviation=None,
                 threshold=0.01,
                 sky_val=0,
                 weighted=False,
                 offset=0):
                 
    """
    
    Create a astropy.nddata.CCDData object for combination.
    
    Parameters
    ----------
    frame: str
        Frame path.
    deviation: str, optional, default None
        Deviation map path.
    threshold: float or np.ndarray, optional
        Threshold for deviation. Can be a map or a value.
    sky_val: float, optional, default 0
        A glocal sky value added for downweighting.
    weighted: bool, optional, default False
        Apply weighting using deviation map if True.
        Apply masking instead if False.
    offset: float, optional, default 0
        A glocal sky value to be subtracted.
        
    Returns
    -------
    ccd: astropy.nddata.CCDData
    
    """
    
    with fits.open(frame) as hdul:
        data = hdul[0].data
        mask = np.isnan(data)
        
        # Add global sky before apply correction
        data = data + sky_val
        
        if deviation is not None:
            dev_map = fits.getdata(deviation)
            
            # Mask region with large deviation
            dev_map[abs(dev_map-1)>threshold] = np.nan
            
            if weighted:
                # Apply weighting using dev_map
                data = data / dev_map
            
            mask = np.isnan(data) | np.isnan(dev_map)
            
            del dev_map
            
        hdul.close()
        
        # Remove global sky after correction
        data = data - sky_val
        
        # Subtract global Offset
        data = data - offset
    
    ccd = CCDData(data, mask=mask, unit=u.adu)
    del mask
        
    return ccd

def get_coverage(frame_list, mask=None): 
    """ Get coverage of stacked exposures. Frames need to be in common wcs. """
    
    shape = fits.getdata(frame_list[0]).shape
    coverage = np.zeros(shape)
    
    for frame in frame_list:
        with fits.open(frame) as hdul:
            data = hdul[0].data
            mask = np.isnan(data)
            coverage[~mask] += 1
            hdul.close()
        
    return coverage


def collect_deviation_maps(dev_paths, color='gray'):
    """ Collect deviation maps from the frame path list """
    dev_maps = np.array([])
    for fn in tqdm(frame_paths):
        with fits.open(fn) as hdul:
            for hdu in hdul:
                dev_map = hdu.data
                dev_maps = np.append(dev_maps, dev_map[~np.isnan(dev_map)])
            del dev_map
            del hdu.data
            hdul.close()

    mean_dev, std_dev = np.mean(dev_maps), mad_std(dev_maps)
    print("Mean/stddev of deviation", mean_dev, std_dev)
    
    plt.hist(dev_maps, color=color)
    plt.axvline(1+3*std_dev, ls='--', color='k')
    plt.axvline(1-3*std_dev, ls='--', color='k')
    
    return dev_maps, mean_dev, std_dev
