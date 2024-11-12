import os
import time
import itertools
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clip, SigmaClip, mad_std
from astropy.modeling import (models, fitting, Fittable2DModel, Parameter)

from skimage.transform import resize

from astropy.utils.exceptions import AstropyUserWarning
import warnings

from reproject import reproject_interp

from .io import redirect_path, move_dir
from .dust import PlanckImage
        
### Sky + Dust Background Modeling

class Worker:
    """ A worker and container for the background modeling. """
    
    def __init__(self, path_or_pixels, wcs=None,
                 PLA_map=None, mask=None):
                 
        """
        Parameters
        ----------
        path_or_pixels : str or ndarray
            An image file name or numpy array of its pixels.
        wcs: astropy.wcs.wcs
            Astropy wcs. If None, needs to be in the file header.
        PLA_map: str or dfcirrus.sky.dust.PlanckImage
            Planck dust model class or path to the map.
        mask: np.ndarray, optional
            Mask map.
        """
        
        
        # Read the fits file by path
        if type(path_or_pixels) == str or type(path_or_pixels) == np.str_:
            self.file_path = path_or_pixels

            # Get data and header
            with fits.open(path_or_pixels) as hdul:
                self.data = hdul[0].data
                self.header = hdul[0].header
                self.wcs = WCS(self.header, relax=True)
                hdul.close()
        else:
            self.data = path_or_pixels
            self.file_path = None
            self.header = None
            if wcs is None:
                raise Exception('Array is given but WCS is missing!')
        
        self.mask = mask
        self.shape = self.data.shape
        
        if wcs is not None:
            self.wcs = wcs

        # Read Planck dust model map from string or directly
        if isinstance(PLA_map, str):
            self.pla_map_path = PLA_map
            self.pla_map = PlanckImage(PLA_map)
        elif isinstance(PLA_map, PlanckImage):
            self.pla_map = PLA_map
        else:
            raise Exception('Input Planck map is not in the right form!')
    
    def safe_delattr(self, attrname):
        if hasattr(self, attrname):
            delattr(self, attrname)
        
    def downsample_wcs(self, scale=0.25):
        """ Downsample WCS. """
        from .utils import downsample_wcs
        wcs_ds = downsample_wcs(self.wcs, scale=scale)
        self.wcs_ds = wcs_ds
        return wcs_ds
        
    def initialize(self,
                   model='radiance',
                   scale=0.25,
                   sn_thre=2,
                   b_size=128):
        """
        Initialization for sky modeling.
        1. downsampling
        2. reproject dust maps
        
        Parameters
        ----------
        scale: float, opional, default 0.25
            Downsampling scale.
        model: 'radiance' or 'tau'
            Planck dust model in use.
        scale: float, opional, default 0.25
            Downsampling scale.
        sn_thre: float, opional, default 2
            SNR threshold for source masking.
        b_size: int, opional, default 128
            Box size used for extracting background.
        """
        
        self.scale = scale
        
        wcs_ds = self.downsample_wcs(scale)
        
        # Get mask map and evaluate small-scale background
        mask_bright, bkg = make_mask_map(self.data, sn_thre=sn_thre, b_size=b_size)
        
        self.bkg = bkg
        
        if self.mask is not None:
            bkg[self.mask] = np.nan
            
        # Downsample small-scale background
        shape_ds = int(self.shape[0]*scale), int(self.shape[1]*scale)
        self.shape_ds = shape_ds
        
        self.bkg_ds = resize(bkg, output_shape=shape_ds, order=3)
        self.sky_val = np.nanmedian(self.bkg_ds)

        # Retrieve Planck dust radiance map
        planck_dust_map = self.pla_map.reproject(wcs_ds, shape_ds, model=model)
        
        # Del the whole map to leave for memory
        self.safe_delattr('pla_map')
        
        # Multiply the model with a factor to maintain float precision
        if model == 'radiance':
            factor = 1e7
        elif model == 'tau':
            factor = 1e5
        
        self.dust_map = planck_dust_map * factor
        
        self.initialized = True
            
    def sky_dust_modeling(self,
                          poly_deg=2,
                          poly_deg_dust=None,
                          model='radiance',
                          scale=0.25,
                          sn_thre=2.5,
                          b_size=128,
                          dust_ratio=None,
                          init_models=None):
        """
        Do Sky + Dust compound background modeling.
        The dust and sky are modelled by 2D polynomials.
        Dust is fitted first and added to the fitting of sky.
        
        Parameters
        ----------
        poly_deg: int, opional, default 2
            Polynomial degree of sky model.
        poly_deg_dust: int, opional, default poly_deg+1
            Polynomial degree of dust model.
        model: 'radiance' or 'tau'
            Planck dust model in use.
        scale: float, opional, default 0.25
            Downsampling scale.
        sn_thre: float, opional, default 2.5
            SNR threshold for source masking.
        b_size: int, opional, default 128
            Box size used for extracting background.
        dust_ratio: float, optional, default None
            Dust ratio fixed if given.
        init_models: dict, optional, default None
            Dictionary which contains two keys.
                'poly_dust_init'
                'compound_init'
            If given, the given model will be used.
        The sky model is stored as self.sky_model.
        """
        
        # Prepare for modeling
        self.initialize(model=model, scale=scale,
                        sn_thre=sn_thre, b_size=b_size)
        
        self.run_fitting(poly_deg=poly_deg,
                         poly_deg_dust=poly_deg_dust,
                         dust_ratio=dust_ratio,
                         init_models=init_models)
        
        # Fetch fitted coefficients
        self.get_coeff_matrix()

    def run_fitting(self,
                    poly_deg=2,
                    poly_deg_dust=None,
                    dust_ratio=None,
                    init_models=None):
    
        """
        Run sky + dust polynomial fitting.
        
        Parameters
        ----------
        poly_deg: int, opional, default 2
            Polynomial degree of sky model.
        poly_deg_dust: int, opional, default poly_deg+1
            Polynomial degree of dust model.
        dust_ratio: float, optional, default None
            Dust ratio fixed if given.
        init_models: dict, optional, default None
            Dictionary which contains two keys.
                'poly_dust_init'
                'compound_init'
            If given, the given model will be used.
        
        """
        
        self.poly_deg = poly_deg
        self.poly_deg_dust = poly_deg_dust
        
        if self.initialized is False:
            raise Exception('Initialization has not been runned!')
            
        dust_map = self.dust_map
        
        # Downsampled Meshgrid
        bkg_ds = self.bkg_ds
        isfinite = np.isfinite(bkg_ds)
        yy_ds, xx_ds = np.indices(bkg_ds.shape)
        
        if dust_ratio==0:
            # Dust model will not be included.
            # Equivalent to common polynomial sky model.
            dust_model = np.zeros_like(self.bkg_ds)
        else:
            if init_models is not None:
                p_dust_init = init_models['poly_dust_init']
            else:
                # Fit dust map with n-th polynomial model
                if poly_deg_dust is None:
                    poly_deg_dust = poly_deg + 1
                p_dust_init = models.Polynomial2D(degree=poly_deg_dust)
            
            self.p_dust_init = p_dust_init
            
            # Run fitting on dust map
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', AstropyUserWarning)
                fit_p_dust = fitting.LevMarLSQFitter()
                p_dust = fit_p_dust(p_dust_init, xx_ds, yy_ds, dust_map)
            
            self.p_dust = p_dust
            
            # Evaluate on the grid
            dust_model = p_dust(xx_ds, yy_ds)
            
        self.dust_model = dust_model

        # Define a custom model
        class Dust_Model(Fittable2DModel):
            amp = Parameter(default=10)

            @staticmethod
            def evaluate(x, y, amp):
                return amp * dust_model[isfinite]

        if init_models is not None:
            m_init = init_models['compound_init']
        else:
            # Initialize poly sky + dust compound model for fitting
            Sky_Model = models.Polynomial2D(degree=poly_deg, c0_0=self.sky_val)
            m_init = Sky_Model + Dust_Model()
            m_init.amp_1.min = 0
            if dust_ratio is not None:
                m_init.amp_1 = dust_ratio
                m_init.amp_1.fixed = True
            #m_init.c0_0_0.min = 0

        self.compound_init = m_init

        # Run fitting on data
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', AstropyUserWarning)
            fit_m = fitting.LevMarLSQFitter()
            m_comp = fit_m(m_init, xx_ds[isfinite], yy_ds[isfinite], bkg_ds[isfinite])

        # m_comp is the mapping function
        self.model_compound = m_comp

        # Evaluate sky models at the meshgrid
        sky_poly = m_comp.left(xx_ds, yy_ds)
        sky_model = resize(sky_poly, output_shape=self.shape, order=3)

        self.sky_model = sky_model

        self.fitting_runned = True
    
    def get_coeff_matrix(self):

        """
        Reshape fitted polynomial coefficients into maxtrix form.
        """
        
        if self.fitting_runned is False:
            raise Exception('Sky model fitting has not been runned!')
        
        deg = self.poly_deg
        
        ij = itertools.product(range(deg+1), range(deg+1))
        ij = np.array(list(ij))
        
        coeff_matrix = np.zeros([deg+1, deg+1])

        for (i, j) in ij:
            coeff = getattr(self.model_compound.left, f'c{i}_{j}', None)
            if coeff!=None:
                coeff_matrix[i,j] = coeff.value
                
        self.sky_coeff = coeff_matrix

         
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
