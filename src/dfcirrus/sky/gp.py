import re
import os
import gc
import sys
import glob
import time
import shutil
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clip, SigmaClip, mad_std
import astropy.units as u

from photutils.background import SExtractorBackground
    
from tqdm import tqdm

from .misc import get_sky_median
from .io import get_segm_name, save_sky_stats, load_sky_stats
from .measurment import *


### Sky Control ###

class container:
    """ A container storing the information and training sample values. """
    
    def __init__(self, frames,
                 wcs_coadd, shape_coadd,
                 pixel_scale=2.85,
                 target='',
                 filt='R',
                 label='',
                 segm_path=None, 
                 sky_path=None):
        
        """
        Parameters
        ----------
        frames: str list
            List of frame paths. Frames must be registered.
        wcs_coadd: astropy.wcs.WCS
            wcs of the registered frames.
        shape_coadd: tuple
            Shape of the coadd.
        pixel_scale: float, optional, default 2.85
            Pixel size in arcsec.
        target: str, optional, default ''
            Target name.
        filt: 'R', or 'G'
            Filter name for Dragonfly.
        label: str, optional, default ''
            Label for file saving.
        segm_path: str, optional, default None
            Directory to read segmentation maps.
        sky_path: str, optional, default None
            Directory to read sky maps.
        """
        
        self.frames = frames
        
        self.wcs = wcs_coadd
        self.shape = shape_coadd
        self.pixel_scale = pixel_scale
        
        self.target = target
        self.filt = filt
        self.label = label
        
        self.segm_path = segm_path
        self.sky_path = sky_path
        
    def generate_landmarks(self, N_landmarks=500, random_state=123456):
        """ Generate random landmarks in the footprint. """
        # Calculate the footprint from wcs, note naxis is in image coordinates (inverted of numpy)
        self.footprint = self.wcs.calc_footprint(axes=self.shape[::-1])
        
        # Generate random landmarkpoints
        self.landmarks_world = generate_landmark_points(self.footprint, size=N_landmarks, random_state=random_state)
        self.landmarks_pix = self.wcs.all_world2pix(self.landmarks_world, 0)
        self.N_landmarks = len(self.landmarks_world)
        
    def get_sky_val(self):
        """ Measure global sky median. """
        sky_vals = np.array([get_sky_median(frame, get_segm_name(frame, self.segm_path)) for frame in self.frames])
        self.sky_vals = sky_vals
        return sky_vals
        
    def measure_local_sky(self, load_pkl='', box_size=512, parallel=True):
        """ Measure local sky value and rms by frames. """
        
        if os.path.exists(load_pkl):
            sky_stats, landmarks_world = load_sky_stats(load_pkl)
            if not np.all(landmarks_world==self.landmarks_world):
                raise Exception('Landmarks do not match!')
        else:
            print(f'Measuring local sky values in {self.filt}-band using box [{box_size}x{box_size}]')
            
            sky_stats = measure_sky_stats_pipe(self.frames, 
                                               self.landmarks_world, 
                                               self.segm_path, 
                                               self.sky_path, 
                                               box_size=[box_size, box_size],
                                               parallel=parallel)

            # Measure global sky mean
            sky_means = self.get_sky_val()

            # Add gloal sky mean to sky measurements
            sky_stats[:,:,0] += np.repeat(sky_means[:,None], self.N_landmarks, axis=1)
            
            # Store sky measurements for future use
            fn = f'{self.target}_{self.filt}_sky_stats_{box_size}{self.label}.pkl'
            save_sky_stats(fn, sky_stats, self.landmarks_world)
        
        self.sky_stats = sky_stats
        
    def generate_training_samples(self, random_state=1):
        """ Generate training samples for background modeling """
        if hasattr(self, 'sky_stats'):
            X, y, Ind, noise = generate_training_samples(self.sky_stats, 
                                                         self.landmarks_pix, 
                                                         random_state=random_state)
            self.X = X
            self.y = y
            self.Ind = Ind
            self.y_noise = noise
            self.trained = True
        else:
            raise Exception('Missing sky measurements!')
            
    def run_GP_modeling(self, scale=1800, n_splits=5):
        """ Run Gaussian process regression to predict a mean sky """
        scale_length = scale/self.pixel_scale
        self.scale_length = scale_length
        
        if getattr(self, 'trained', False):
            gp_list = gaussian_process_regression(self.X, self.y, self.y_noise, n_splits=n_splits,  
                                                  scale_length=scale_length, wcs=self.wcs, Ind=self.Ind)
            self.gp_list = gp_list
        else:
            raise Exception('Missing training samples!')
        
    def generate_deviation_maps(self, save_path='./', perturb=1, random_state=1):
        """ Generate deviation maps by frames based on the model """
        if hasattr(self, 'gp_list'):
            generate_deviations_by_frames(self.frames, 
                                          self.X, 
                                          self.y, 
                                          self.Ind, 
                                          self.gp_list, 
                                          self.shape,
                                          scale_length=self.scale_length, 
                                          perturb=perturb, 
                                          random_state=random_state, 
                                          save_path=save_path)
            print('Done!')
        else:
            raise Exception('Missing GP model!')

    def run_sky_modeling(self, load_pkl='', 
                         box_size=512, 
                         scale=3600, 
                         parallel=True, 
                         generate_deviation=False,
                         save_path='./'):
        """
        
        Run GPR sky modeling.
        
        load_pkl: str
            Filename of the measurement pkl to read.
        box_size: 1x2 array, default [512, 512]
            Size of the box estimator.
        scale_length: float
            Scale length in arcsec of the kernel.
        parallel: bool
            If True, run GPR sky modeling in parallel.
        generate_deviation: bool
            If True, generate and save deviation maps.
        save_path: str
            Directory to save deviation maps.
            
        """
        
        self.measure_local_sky(load_pkl, box_size=box_size, parallel=parallel)
        self.generate_training_samples()
        self.run_GP_modeling(scale)
        
        if generate_deviation:
            self.generate_deviation_maps(save_path=save_path)
        
    
def generate_training_samples(sky_stats, landmarks_pix, random_state=1):
    """
    
    Generate trainging samples for GP regression from the measurements.
    The target values and noises are normalized by the global median value
    of the frame.
    Extreme values (>10 sigma) are excluded.
    
    Parameters
    ---------- 
    sky_stats: MxNx2 array
       Local sky measurements (sky, rms) of N landmarks of M frames.
       
    landmarks_pix: Nx2 array
        Pixel coordinates of N landmarks (RA, Dec).
   
    Returns
    -------
    X: Nx2 array
        Variables for training (positions of landmarks).
    
    y: MxN array
        Target values for training (normalized local background value)
    
    Ind: MxN array
        Indice stores the index of the frame in the list in the arrays.
    
    noise: MxN array
        Noise of y for training (normalized local background rms)
    
    Returns
    -------
    X: Nx2 array
        Variables for training (positions of landmarks).
    
    y: MxN array
        Target values for training (normalized local background value)
    
    noise: MxN array
        Noise of y for training (normalized local background rms)
        
    """
    
    np.random.seed(random_state)
    
    N = len(sky_stats)
    
    # First axis of sky_stats is mean value. Take median along frames.
    sky_norms = np.nanmedian(sky_stats[:,:,0],axis=1)
    
    # Correct sky_norms by the frame mean
    sky_means = np.nanmedian(np.array([bkg_vals/sky_norm for (bkg_vals, sky_norm) in zip(sky_stats[:,:,0], sky_norms)]),axis=0)
    sky_norms = np.nanmedian((sky_stats[:,:,0]/sky_means),axis=1)
    
    # Convert to image coordinates
    landmarks_X, landmarks_Y = landmarks_pix.T

    for i in range(N):
        # landmark positions
        X = np.vstack([X, landmarks_pix]) if i!=0 else landmarks_pix
        # index of frame
        Ind = np.hstack([Ind, np.ones(len(landmarks_pix))*(i)]) if i!=0 else np.ones(len(landmarks_pix))*(i)

    X += np.random.normal(0, 1, size=X.shape)
    
    # Normalized background values
    y = np.concatenate([bkg_vals/sky_norm for (bkg_vals, sky_norm) in zip(sky_stats[:,:,0], sky_norms)])
    y_noise = np.concatenate([bkg_rms/sky_norm for (bkg_rms, sky_norm) in zip(sky_stats[:,:,1], sky_norms)])
    
    # Exclude extreme data points
    y_std = mad_std(y[~np.isnan(y)])
    print('y stddev = {:.4f}'.format(y_std))
    y_q1, y_q2 = 1-10*y_std, 1+10*y_std
    good = (~np.isnan(y)) &  (y > y_q1) &  (y < y_q2)
    X, y, Ind, noise = X[good], y[good], Ind[good], y_noise[good]
    
    return X, y, Ind, noise


def gaussian_process_regression(X, y, noise, 
                                scale_length=1800/2.85,
                                pixel_scale=2.85,
                                n_splits=5,
                                max_n_samples=10000,
                                fix_scale_length=True,
                                random_state=1,
                                wcs=None, Ind=None,
                                ra_quantile=0.5,
                                dec_quantile=0.5,
                                dX_range=200,
                                dY_range=200, plot=True):
    
    """ 
    Gaussian regression modeling of sky using landmarks. 
    
    Parameters
    ----------
    X: Nx2 np.ndarray
        Variables for training (positions of landmarks).
    
    y: MxN np.ndarray
        Target values for training (normalized local background value)
    
    noise: MxN array
        Noise of y for training (normalized local background rms)
    
    scale_length: float
        Scale length in pixel of the kernel.
        If fix_scale_length=False, this is the initial guess.
        
    n_splits: int, optional, default 5
        K-fold number for cross-validation
        
    max_n_samples: int, optional, default 10000
        Maximum number of sample in each fold
    
    fix_scale_length: bool, optional, default True
        Whether to fix scale length in the training.
        
    wcs: astropy.wcs.WCS, optional
        WCS of landmarks to transform the unit in plotting.
    
    Ind: MxN np.ndarray, default None
        Indice stores the index of the frame in the list in the arrays.
        This is used to connect points from the same frame.
    
    ra_quantile: float, opional, default 0.5
        Quantile at which the Dec slice is displayed for viz.
        
    dec_quantile float, opional, default 0.5
        Quantile at which the RA slice is displayed for viz.
        
    dX_range: float, opional, default 200
        X range in pixel at which the RA slice is displayed for viz.
        
    dY_range: float, opional, default 200
        Y range in pixel at which the Dec slice is displayed for viz.
        
    plot: bool, optional, default True
        If True, plot modeling result in X/Y slice for viz.
    
    Returns
    -------
    gp_list: list of sklearn.gaussian_process.GaussianProcessRegressor
        List of Gaussian Process Regressor
        
    """
    from sklearn.gaussian_process import GaussianProcessRegressor, kernels
    from sklearn.model_selection import KFold

    gp_list = []
    
    n_splits = max(n_splits, len(X)//max_n_samples)
    if len(X)<1500: n_splits = 3
        
    kf = KFold(n_splits, shuffle=True, random_state=random_state) # w/ shuffle
    
    print('Training with %d-fold GP...'%n_splits)
    
    if fix_scale_length==True:
        length_scale_bounds = 'fixed'
        print('Scale length is fixed.')
    else:
        length_scale_bounds = (1e1, 1e4)
    kernel = kernels.RBF(scale_length, length_scale_bounds)
    
    for remain_ind, fold_ind in kf.split(X, y):
        X_train, y_train, noise_train = X[fold_ind], y[fold_ind], noise[fold_ind]
        gp = GaussianProcessRegressor(kernel=kernel, alpha=noise_train**2, n_restarts_optimizer=8, random_state=random_state)
        gp.fit(X_train, y_train)
        gp_list.append(gp)
        if fix_scale_length==False:
            print("GP Kernel Scale Lenght in pix: ", gp.kernel.length_scale)
    
    if plot:
        # Build grids for plotting
        fig, (ax1,ax2) = plt.subplots(1,2, figsize=(16,6),dpi=100)
        
        # quantile of max/min
        X_min, X_max = np.quantile(X[:,0], [0.001, 0.999]).astype(int)
        Y_min, Y_max = np.quantile(X[:,1], [0.001, 0.999]).astype(int)
        
        # Build grids for plotting
        Nmesh_X, N_mesh_Y = (X_max-X_min)//100,  (Y_max-Y_min)//100
        
        grid_pred = np.array([[x, y] for x in np.linspace(X_min,X_max,Nmesh_X) for y in np.linspace(Y_min,Y_max,N_mesh_Y)])
        
        # Run prediction on grids
        result_pred = np.array([np.transpose(gp.predict(grid_pred, return_std=True)) for gp in gp_list])
        y_pred, sigma = np.median(result_pred, axis=0).T
        
        sigma = sigma/n_splits
                
        # A narrow slice for visualization
        Y_check = np.nanquantile(X[:,1], dec_quantile)
        X_check = np.nanquantile(X[:,0], ra_quantile)
        
        cond_Y = lambda x_: abs(x_[:,1]-Y_check) < dY_range//2
        cond_X = lambda x_: abs(x_[:,0]-X_check) < dX_range//2
        
        # Grid for plotting smooth prediction
        gg_x = np.array([[x, Y_check] for x in range(X_min,X_max,50)])
        gg_y = np.array([[X_check, y] for y in range(Y_min,Y_max,50)])
        
        if wcs is not None:
            # convert to ra, dec
            X_world = wcs.all_pix2world(X, 0)
            X_check_world, Y_check_world = wcs.all_pix2world([[X_check, Y_check]], 0)[0]
            ra_range = 2*abs(wcs.all_pix2world([[X_check+dX_range//2, Y_check]], 0)[0][0] - X_check_world)
            dec_range = 2*abs(wcs.all_pix2world([[X_check, Y_check+dY_range//2]], 0)[0][1] - Y_check_world)
            
            ax1.set_xlabel('RA', fontsize=18)
            ax2.set_xlabel('Dec', fontsize=18)
            ax1.text(0.1, 0.85, 'Dec = %.2f $+/-$ %.2f'%(Y_check_world, dec_range/2.), transform=ax1.transAxes, fontsize=18)
            ax2.text(0.1, 0.85, 'RA = %.2f $+/-$ %.2f'%(X_check_world, ra_range/2.), transform=ax2.transAxes, fontsize=18)
        else:
            ax1.set_xlabel('X')
            ax2.set_xlabel('Y')
            ax1.text(0.1, 0.85, 'Y = %d +/- %d'%(Y_check, Y_range), transform=ax1.transAxes, fontsize=14)
            ax2.text(0.1, 0.85, 'X = %d +/- %d'%(X_check, X_range), transform=ax2.transAxes, fontsize=14)
    
        # Plot results in X/Y slice
        for k, (cond, gg, ax) in enumerate(zip([cond_Y, cond_X],[gg_x, gg_y], [ax1,ax2])):
            
            # slice condition on grid
            cond_grid = cond(grid_pred)
            cond_train = cond(X)
        
            if wcs is not None:
                grid_pred = wcs.all_pix2world(grid_pred, 0)
            
            grid_slice = grid_pred[cond_grid][:,k]
            
            # predicted mean
            result_gg = np.array([np.transpose(gp.predict(gg, return_std=True)) for gp in gp_list])
            yy_gg, sigma_gg = np.median(result_gg, axis=0).T
            sigma_gg = sigma_gg/n_splits
            y_pred_min = np.min(result_gg[:,:,0], axis=0)
            y_pred_max = np.max(result_gg[:,:,0], axis=0)
            
            if wcs is not None:
                # convert to ra, dec
                gg_world = wcs.all_pix2world(gg, 0)
            
            ax.fill_between(gg_world[:,k], yy_gg - 1.96 * sigma_gg, yy_gg + 1.96 * sigma_gg,
                            alpha=.2, fc='firebrick', ec='None', label='Pred. 95% C.I.')
            ax.fill_between(gg_world[:,k], y_pred_min, y_pred_max, edgecolor='none', facecolor='r', lw=0, alpha=0.4, label='Pred. Mean Span')
            
            #Scatter plot of landmarks
            ax.scatter(X_world[:,k][cond_train], y[cond_train], s=20, alpha=0.3, linewidths=1, facecolors='k', edgecolors='gold')
            
            if Ind is not None:
                # connect landmarks by frames
                for ind in np.unique(Ind):
                    use_ind = cond_train & (Ind==ind)
                    sortby = np.argsort(X[:,k][use_ind])
                    ax.plot(X_world[:,k][use_ind][sortby], y[use_ind][sortby], lw=0.2, color='k', alpha=0.1)
            ax.plot(gg_world[:,k], yy_gg, lw=3.5, color='gold', alpha=0.9, zorder=6, label='Pred. Mean',
                     path_effects=[pe.Stroke(linewidth=5.5, foreground='k', alpha=0.8), pe.Normal()])
            ax.set_ylim(0.95,1.05)
            if k==0:
                ax.set_xlim(gg_world[:,0].max()+0.5, gg_world[:,0].min()-0.5)
            else:
                ax.set_xlim(gg_world[:,1].min()-0.1,gg_world[:,1].max()+0.1)
                            
            lgnd = ax.legend(fontsize=18, loc=4, handletextpad=1)
        
    return gp_list

def generate_deviations_by_frames(frames, X, y, Ind, gp_list, shape,
                                  scale_length=1800/2.85, perturb=1, 
                                  save_path='./', random_state=1):
    
    """ 
    Generate deviation maps of frames. The deviation map is the difference map
    in percentile relative to the median background interpolated from landmarks.
    
    Parameters
    ----------    
    frames: list
        List of frame paths.
        
    X: Nx2 array
        Variables for training (positions of landmarks).
    
    y: MxN array
        Target values for training (normalized local background value)
    
    Ind: MxN array
        Indice stores the index of the frame in the list in the arrays.
    
    noise: MxN array
        Noise of y for training (normalized local background rms)
    
    gp_list: list of sklearn.gaussian_process.GaussianProcessRegressor
        List of Gaussian Process Regressor
        
    shape: 1x2 tuple
        Image shape.
        
    scale_length: float
        Scale length of the RBF kernel for GP training in pixel unit.
        
    perturb: int, default 1
        Perturbation of landmarks in pixel on the input sample.
    
    save_path: str
        Path to save deviation maps.
    
    """
    
    from scipy.interpolate import griddata
    from sklearn import preprocessing
    from sklearn.gaussian_process import GaussianProcessRegressor, kernels
    
    np.random.seed(random_state)
    
    # Calculate mean predicted local sky values (averaged by frames) and local deviations at landmarks
    X_prtb = (X + np.random.normal(0, perturb, size=X.shape)).astype(int)
    print('Computing mean predicted background at landmarks...')
    y_pred = np.array([gp.predict(X_prtb) for gp in gp_list]).mean(axis=0)
    
    A = y / y_pred

    # Grids for generating deviation maps
    grid_X, grid_Y = np.mgrid[1:shape[1]+1, 1:shape[0]+1]
    dX_sp, dY_sp = shape[1]//40, shape[0]//40
    grid_sparse = np.array([[x, y] for x in np.arange(1,shape[1]+1,dX_sp) for y in np.arange(1,shape[0]+1,dY_sp)])  # Sparse grid
    grid = np.array([[x, y] for x in np.arange(1,shape[1]+1,1) for y in np.arange(1,shape[0]+1,1)])  # Full grid
    
    # Preprocessing for grid points
    scaler = preprocessing.MinMaxScaler().fit(X)
    kernel = kernels.RBF(scale_length/scaler.data_range_, 'fixed')
    
    print('Generating Deviation Maps...')
        
    for ind, frame in enumerate(tqdm(frames)):
        header = fits.getheader(frame)
        name = os.path.basename(frame).split('_light')[0]
        use_ind = (Ind==ind)
        X_ind = X[use_ind]
        A_ind = A[use_ind]
        X_scaled = scaler.transform(X_ind)

        # GP regression
        gp_ind = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=8)
        gp_ind.fit(X_scaled, A_ind)

        # GP fit
        grid_scaled = scaler.transform(grid_sparse)
        A_grid_sparse = gp_ind.predict(grid_scaled, return_std=False)
        
        # Interpolation
        grid_zg = griddata(grid_sparse, A_grid_sparse, (grid_X, grid_Y), method='cubic').T

        # Remove extrapolation
        grid_z1 = griddata(X_ind, A_ind, (grid_X, grid_Y), method='linear').T
        grid_zg[np.isnan(grid_z1)] = np.nan

        fn_dev = os.path.join(save_path, name+'_dev.fits')
        fits.writeto(fn_dev, data=grid_zg, header=header, overwrite=True)
