import os
import time
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clip, SigmaClip, mad_std

from astropy.utils.exceptions import AstropyUserWarning
import warnings

from reproject import reproject_interp

from .io import redirect_path, move_dir
from .dust import PlanckImage
        
### Sky + Dust Background Modeling

class Worker:
    """ A worker and container for the background modeling. """
    
    def __init__(self, frame, PLA_map=None, mask=None):
                 
        """
        Parameters
        ----------
        frame: str
            Frame path.
        PLA_map: str
            Path of Planck dust model.
        mask: np.ndarray, optional
            Mask map.
        """
                 
        self.file_path = frame

        # Get data and header
        with fits.open(frame) as hdul:
            self.data = hdul[0].data
            self.header = hdul[0].header
            hdul.close()
        
        self.mask = mask

        self.shape = self.data.shape

        self.wcs = WCS(self.header, relax=True)

        # Read Planck dust model map
        self.pla_map = PlanckImage(PLA_map)

    def downsample_wcs(self, scale=0.25):
        """ Downsample WCS. """
        from .utils import downsample_wcs
        wcs_ds = downsample_wcs(self.wcs, scale=scale)
        return wcs_ds

    def sky_dust_modeling(self,
                          poly_deg=2,
                          scale=0.25,
                          model='radiance',
                          poly_deg_dust=None,
                          sn_thre=2, b_size=128):
        """ 
        Do Sky + Dust compound background modeling.
        
        Parameters
        ----------
        poly_deg: int, opional, default 2
            Polynomial degree of sky model.
        scale: float, opional, default 0.25
            Downsampling scale.
        model: 'radiance' or 'tau'
            Planck dust model in use.
        poly_deg_dust: int, opional, default poly_deg+1
            Polynomial degree of dust model.
        sn_thre: float, opional, default 2
            SNR threshold for source masking.
        b_size: int, opional, default 128
            Box size used for extracting background.
            
        The sky model is stored as self.sky_model.
        """
        
        from astropy.modeling import (models, fitting, Fittable2DModel, Parameter)
        from skimage.transform import resize

        self.scale = scale
        
        wcs_ds = self.downsample_wcs(scale)
        self.wcs_ds = wcs_ds
        
        # Get mask map and evaluate small-scale background
        mask_bright, bkg = make_mask_map(self.data, sn_thre=sn_thre, b_size=b_size)
        
        self.bkg = bkg
        
        if self.mask is not None:
            bkg[self.mask] = np.nan
            
        # Downsample small-scale background
        shape_ds = int(self.shape[0]*scale), int(self.shape[1]*scale)
        self.shape_ds = shape_ds
        
        bkg_ds = resize(bkg, output_shape=shape_ds, order=3)
        sky_val = np.nanmedian(bkg_ds)
        
        self.bkg_ds = bkg_ds

        # Retrieve Planck dust radiance map
        dust_model_map = self.pla_map.reproject(wcs_ds, shape_ds, model=model)
        
        # Multiply the model with a order-of-magnitude factor
        if model == 'radiance':
            factor = 1e7
        elif model == 'tau':
            factor = 1e5
        
        dust_map = dust_model_map * factor
            
        self.dust_map = dust_map
        
        # Downsampled Meshgrid
        isfinite = np.isfinite(bkg_ds)
        yy_ds, xx_ds = np.indices(bkg_ds.shape)
        
        # Fit dust map with n-th polynomial model
        if poly_deg_dust is None:
            poly_deg_dust = poly_deg + 1
        p_dust_init = models.Polynomial2D(degree=poly_deg_dust)
        fit_p_dust = fitting.LevMarLSQFitter()
        p_dust = fit_p_dust(p_dust_init, xx_ds, yy_ds, dust_map)
        
        # Evaluate on the grid
        dust_model = p_dust(xx_ds, yy_ds)
        self.dust_model = dust_model

        # Define a custom model
        class Dust_Model(Fittable2DModel):
            amp = Parameter(default=1)

            @staticmethod
            def evaluate(x, y, amp):
                return amp * dust_model[isfinite]
        
        # Initialize poly sky + dust compound model for fitting
        m_init = models.Polynomial2D(degree=poly_deg, c0_0=sky_val) + Dust_Model()
        m_init.amp_1.min = 0
        #m_init.c0_0_0.min = 0

        # Run fitting
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', AstropyUserWarning)
            fit_m = fitting.LevMarLSQFitter()
            m_comp = fit_m(m_init, xx_ds[isfinite], yy_ds[isfinite], bkg_ds[isfinite])
        
        # m_comp is the mapping function
        self.model_compound = m_comp
        
        # Evaluate dust and sky models at the meshgrid
        dust_model = dust_model * m_comp.amp_1
        sky_poly = m_comp.left(xx_ds, yy_ds)
        sky_model = resize(sky_poly, output_shape=self.shape, order=3)

        self.sky_model = sky_model
        
                
### Dragonfly pipeine specfic helper function ###

def sky_dust_modeling_by_frame(master_light,
                               PLA_map,
                               sky_subtract_path,
                               sky_path=None,
                               save_sky=True):
                        
    """
    Run sky+dust background modeling by frame. Sky-subtracted images will be
    saved in sky_subtract_path. Sky models are saved to sky_path if True.
    
    Parameters
    ----------
    master_lights: list of str
        List of frame paths of the master lights.
    PLA_map: str
        Path of Planck dust model.
    sky_subtract_path: str
        Directory to save sky-subtracted frames.
    sky_path: str, optional, default None
        Directory to save sky models.
    save_sky: bool, optional, default True
        Whether to save sky models.
        
    """
    
    basename = os.path.basename(master_light)

    # Dragonfly specific name
    fn_light_ss = os.path.join(sky_subtract_path, basename.replace('_light', '_light_ss'))
    fn_sky = os.path.join(sky_path, basename.replace('_light', '_light_bg'))

    # Create a worker class for the modeling
    worker = Worker(frame, PLA_map)
    data = worker.data
    header = worker.header

    # Run sky + dust modeling
    worker.sky_dust_modeling()
    
    sky_model = worker.sky_model

    # Subtract sky model
    data_ss = data - sky_model

    header['BKGVAL'] = np.nanmedian(sky_model)

    # Save
    fits.writeto(fn_light_ss, data=data_ss, header=header, overwrite=True)
    if save_sky & (sky_path!=None):
        fits.writeto(fn_sky, data=sky_model, header=header, overwrite=True)

def sky_dust_modeling_pipeline(master_lights,
                               PLA_map,
                               sky_subtract_path,
                               sky_path=None,
                               save_sky=True,
                               work_dir='./', 
                               N_cores=4,
                               parallel=True):

    """ 
    A wrapped pipeline function for subtracting large-scale background
    using polynomial sky + dust compound models.
    
    Sky-subtracted images and sky models will be saved locally and then
    moved to the designated path.

    Parameters
    ---------- 
    master_lights: list of str
        List of frame paths of the master lights.
    PLA_map: str
        Path of Planck dust model.
    sky_subtract_path: str
        Directory to save sky-subtracted frames.
    sky_path: str, optional, default None
        Directory to save sky models.
    save_sky: bool, optional, default True
        Whether to save sky models.
    N_cores: int, optional, default 4
        Number of cores in use if parallel=True
    parallel: bool, optional, default True
        Whether to run sky modeling in parallel.

    """
    
    print('Save background-subtracted frames to ', sky_subtract_path)
    print('Save background to ', sky_path)
    
    kwargs = dict(PLA_map=PLA_map,
                  sky_subtract_path=sky_subtract_path,
                  sky_path=sky_path,
                  save_sky=save_sky,
                  work_dir=work_dir)
    
    if parallel:
        from functools import partial
        from .parallel import parallel_compute
        
        p_sky_modeling = partial(sky_dust_modeling_by_frame, **kwargs)
        parallel_compute(master_lights, p_sky_modeling, cores=N_cores, 
                         lengthy_computation=True, verbose=True)
    
    else:
        for master_light in tqdm(master_lights):
            sky_dust_modeling_by_frame(master_light, **kwargs)
    
    # Move the outputs from current diectory to designated one.
    for src_path in [sky_subtract_path, sky_path]:
        if src_path != None:
            dst_path = redirect_path(src_path, work_dir)
            print(f'Moving files to {dst_path} ...')
            move_dir(src_path, dst_path)


### Background Extraction function ###

def background_extraction(image, mask=None,
                          return_rms=True,
                          b_size=1024, f_size=3,
                          exclude_percentile=20,
                          maxiters=10):
    """ Extract a 2D background using SExtractor estimator with source masked. """
    from photutils import Background2D, SExtractorBackground
    
    try:
        Bkg = Background2D(image, mask=mask, bkg_estimator=SExtractorBackground(),
                           box_size=b_size, filter_size=f_size, exclude_percentile=exclude_percentile,
                           sigma_clip=SigmaClip(sigma=3., maxiters=maxiters))
        bkg = Bkg.background
        bkg_rms = Bkg.background_rms
        
    except ValueError:
        img = image.copy()
        if mask is not None:
            img[mask] = np.nan
        bkg, bkg_rms = np.nanmedian(image) * np.ones_like(image), np.nanstd(image) * np.ones_like(image)
        
    if return_rms:
        return bkg, bkg_rms
    else:
        return bkg
    
def make_mask_map(image, sn_thre=3,
                  b_size=128, f_size=3,
                  exclude_percentile=20, maxiters=10,
                  npix=5, return_bkg=True):
    """ Make mask map for the image based on local S/N threshold. """
    from photutils import detect_sources, deblend_sources
    
    # Evaluate background and threshold
    mask = ~np.isfinite(image)
    bkg, bkg_rms = background_extraction(image, mask=mask, b_size=b_size, f_size=f_size,
                                         exclude_percentile=exclude_percentile, maxiters=maxiters)
    threshold = bkg + (sn_thre * bkg_rms)
    
    # detect source
    segm0 = detect_sources(image, threshold, npixels=npix)
    segmap = segm0.data.copy()
    
    # Get mask
    mask = (segmap!=0)
    
    if return_bkg:
        return mask, bkg
    else:
        return mask
