import re
import os
import gc
import sys
import time
import shutil
import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clip, SigmaClip, mad_std

from photutils.background import SExtractorBackground
    
from tqdm import tqdm


### Sky Measurement ###
def generate_landmark_points(footprint, size=500, random_state=1):
    """ 
    Generate coordinates of landmarks in the footprint.
    
    Parameters
    ----------
    footprint: 4x2 array
        Positions of the corners of the image on the sky.
    
    size: int
        Number of landmarks.
    
    Returns
    -------
    coords: Nx2 array
        Coordinates of N landmarks (RA, Dec).
    
    """
    
    np.random.seed(random_state)
    
    RA_fp, DEC_fp = footprint[:,0], footprint[:,1]
    ra_mean, dec_mean = footprint.mean(axis=0)
    angle_factor = np.cos(dec_mean/180*np.pi)
    
    coords_min = np.array([min(RA_fp[0:2]), max(DEC_fp[0], DEC_fp[3])])
    coords_max = np.array([max(RA_fp[2:4]), min(DEC_fp[1], DEC_fp[2])])
    
    # Grid density corrected by dec
    coords_range = np.abs(coords_max - coords_min)
    ra_dec_range_ratio = coords_range[0]/coords_range[1] * angle_factor
    Dec_size = int(np.ceil(np.sqrt(size / ra_dec_range_ratio)))
    RA_size = int(np.ceil(Dec_size * ra_dec_range_ratio))
    
    # Grid of points with random jitter
    coords = np.array([[x, y] for x in np.linspace(coords_min[0],coords_max[0],RA_size) for y in np.linspace(coords_min[1],coords_max[1],Dec_size)])
    scale = np.round([coords_range[0]/RA_size/5, coords_range[1]/Dec_size/5], 3)
    coords += np.random.normal(loc=0, scale=scale, size=(len(coords),2))    
    
    return coords

def get_local_sky(landmark_pix, data, mask=None, box_size=[512,512], pad=32):
    """ 
    Get local sky value using a box estimator given the position of the landmark.
    Return nan if the landmark falls out of boundary.
    
    Parameters
    ----------    
    landmark_pix: 1x2 array
        Position of the landmark in pixel coordinates.
    
    data: 2d array
        Image.
    
    box_size: 1x2 array
        Size of the box estimator.
    
    pad: int
        Pad at the footprint edges. Drop if too close.
    
    Returns
    -------
    bkg_val: float
        estimate of local background value
    
    bkg_rms: float
        estimate of local background rms
        
    """
    
    X, Y = landmark_pix
    
    # skip measurement for out-of-boundary landmarks
    oob = (X > data.shape[1] - pad) | (Y > data.shape[0] - pad) | (min(X, Y) < pad)

    if oob:
        return np.nan, np.nan
    else:
        # Range for the cutout around landmarks
        xmin, xmax = int(max(0, Y-box_size[1]//4)), int(min(data.shape[0], Y+box_size[1]//4))
        ymin, ymax = int(max(0, X-box_size[0]//4)), int(min(data.shape[1], X+box_size[0]//4))
        
        # Cutout around landmarks
        data_cutout = data[xmin:xmax, ymin:ymax]

        if mask is not None:
            mask_cutout = mask[xmin:xmax, ymin:ymax]
        else:
            mask_cutout = False
            
        data_ma = np.ma.array(data_cutout, mask=mask_cutout)
        isnan = np.isnan(data_ma)
        
        if data_ma[~isnan].count() < 0.2 * np.prod(box_size):
            return np.nan, np.nan

        bkg = SExtractorBackground(sigma_clip = SigmaClip(sigma=3.0))
        bkg_val = bkg.calc_background(data_ma[~isnan])
        bkg_rms = mad_std(data_ma[~isnan])
        
        del data
        del mask
        del bkg
        del data_cutout
        del mask_cutout
        del data_ma
        gc.collect()
        
        return bkg_val, bkg_rms


def measure_sky_stats_by_frame(frame,
                               landmarks_world,
                               segm_path=None,
                               sky_path=None,
                               box_size=[512,512]):
    """
    Measure sky values and store as .npy at landmark positions.
    Input frame has not have been reprojected.
    
    Parameters
    ----------
    frame: list
        Frame path.
        
    landmarks_world: Nx2 array
        World coordinates of N landmarks (RA, Dec).
        
    box_size: 1x2 array, default [512, 512]
        Size of the box estimator.
        
    segm_path: str, default None
        Path to read segmentation maps as masks. If None, no mask.
        
    sky_path: str, default None
        Path to read sky background maps. If None, no sky subtraction.
    
    Returns
    -------
    sky_stats: MxNx2 array
        Local sky measurements (bkg_val, bkg_rms) of N landamrks of M frames.
    
    """
        
    name = os.path.basename(frame).split('_light')[0]

    if os.path.exists(f'bkg_tmp/{name}_bkg.npy'):
        return False
        
    with fits.open(frame, memmap=True) as hdul:
        for hdu in hdul:
            data = hdu.data
            wcs = WCS(hdu.header)

            landmarks_pix = wcs.all_world2pix(landmarks_world, 0)

            # Read mask map
            if segm_path is not None:
                fn_segm = os.path.join(segm_path, name+'_segm.fits')
                with fits.open(fn_segm, memmap=True) as hdul_seg:
                    for hdu_seg in hdul_seg:
                        segm = hdu_seg.data
                        mask = segm>0
                        del segm
                        del hdu_seg.data
                    hdul_seg.close()
            else:
                mask = np.zeros_like(data, dtype=bool)

            # Subtract sky map
            if sky_path is not None:
                fn_bkg = os.path.join(sky_path, name+'_light_bg.fits')
                bkg = fits.getdata(fn_bkg)
                data = data - bkg
                del bkg
                
            # Get local sky value and rms
            bkg_stats = np.array([get_local_sky(lm, data, mask, box_size=box_size)
                                  for lm in landmarks_pix])[None, :, :]
            
            # Save measurements as local npy file
            np.save(f'bkg_tmp/{name}_bkg.npy', bkg_stats)

            # Clean memory
            del mask
            del data
            del wcs
            del hdu.data
            del bkg_stats
            del landmarks_pix

        hdul.close()
        # Sleep and clear memory
        time.sleep(0.05)
        gc.collect()
        return True


def measure_sky_stats_pipe(frames, 
                           landmarks_world, 
                           segm_path=None, 
                           sky_path=None, 
                           box_size=[512,512],
                           parallel=True):
    """ 
    
    Measure sky values and store as .npy at landmark positions of frames.
    Input frames have not have been reprojected.
    
    Parameters
    ----------     
    frames: list
        List of frame paths.
        
    landmarks_world: Nx2 array
        World coordinates of N landmarks (RA, Dec).
        
    box_size: 1x2 array, default [512, 512]
        Size of the box estimator.
        
    segm_path: str, default None
        Path to read segmentation maps as masks. If None, no mask.
        
    sky_path: str, default None
        Path to read sky background maps. If None, no sky subtraction.
    
    Returns
    -------
    sky_stats: MxNx2 array
        Local sky measurements (bkg_val, bkg_rms) of N landamrks of M frames.
    
    """
    
    if sky_path is None:
        print('No sky subtraction for the frames.')
    else:
        print(f'Sky has not been subtracted for the frames. Do sky subtraction using sky maps in {sky_path}')
    
    if not os.path.exists('bkg_tmp'):
        os.mkdir('bkg_tmp')
    
    kwargs = dict(landmarks_world=landmarks_world, 
                  segm_path=segm_path, 
                  sky_path=sky_path, 
                  box_size=box_size)
        
    if parallel:
        from functools import partial
        from elderflower.parallel import parallel_compute
        start = time.time()
        p_measure_sky_stats = partial(measure_sky_stats_by_frame, **kwargs)
        parallel_compute(frames, p_measure_sky_stats, cores=4, lengthy_computation=True, verbose=True)
        end = time.time()
        print('Total Time: {:.2f}'.format(end-start))
        
    else:
        for frame in tqdm(frames):
            measure_sky_stats_by_frame(frame, **kwargs)
    
    # Collect sky measurements of all frames and combine into one array
    for i, frame in enumerate(frames):
        name = os.path.basename(frame).split('_light')[0]
        bkg_stats = np.load(f'bkg_tmp/{name}_bkg.npy')
        sky_stats = np.vstack([sky_stats, bkg_stats]) if i!=0 else bkg_stats
    
    shutil.rmtree('bkg_tmp')
    return sky_stats
