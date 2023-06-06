import re
import os
import glob
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

from astropy.io import fits
import astropy.units as u

from .measurement import get_local_sky

from astropy.utils.exceptions import AstropyUserWarning
import warnings

def colorbar(mappable, pad=0.2, size="5%", loc="right",
             ticks_rot=None, ticks_size=12, color_nan='gray', **args):
    """ Customized colorbar """

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    
    if loc=="bottom":
        orent = "horizontal"
        pad = 1.5*pad
        rot = 60 if ticks_rot is None else ticks_rot
    else:
        orent = "vertical"
        rot = 0 if ticks_rot is None else ticks_rot
    
    cax = divider.append_axes(loc, size=size, pad=pad)
    
    cb = fig.colorbar(mappable, cax=cax, orientation=orent, **args)
    cb.ax.set_xticklabels(cb.ax.get_xticklabels(),rotation=rot)
    cb.ax.tick_params(labelsize=ticks_size)
    
    #cmap = cb.mappable.get_cmap()
    cmap = copy(plt.cm.get_cmap())
    cmap.set_bad(color=color_nan, alpha=0.3)
    
    return cb


def sky_match_coadd(frames, coverages, filt, indice_pairs, apass_dir='/Volumes/mice/apass/'):
    """ Calculate the sky offset of the input frames. """
    zps, flux_scales = calculzate_flux_scales(frames, filt, apass_dir=apass_dir)
    
    # Scale images and set low coverage to nan
    images = []
    for i, (frame, frame_covergae, flux_scale) in enumerate(zip(frames, coverages, flux_scales)):
        image = fits.getdata(frame)

        # Read weight map
        coverage = fits.getdata(frame_covergae)

        # Set minimum frames required
        N_frame = int(np.nanmax(coverage))
        N_frame_min = np.max([10, N_frame//2])
        image[coverage<N_frame_min] = np.nan

        images.append(image)
    
    # display images
    fig, axes = plt.subplots(3,2, figsize=(8,12))
    for i in range(3):
        for j in range(2):
            ax = axes[i,1-j]
            ax.imshow(images[1+i*2+j]*flux_scales[1+i*2+j], vmin=-20, vmax=50)
    plt.tight_layout()
    plt.show()
    
    # calculates sky match statistics
    bkg_stats_pairs = []
    landmarks_pairs = []
    for ind_pair in indice_pairs:
        bkg_stats_pair, landmarks_pix_pair = calculate_overlap_stats(images, ind_pair[0], ind_pair[1])
        bkg_stats_pairs.append([bkg_stats_pair])
        landmarks_pairs.append(landmarks_pix_pair) 
        
    delta_array = calculate_sky_offsets(bkg_stats_pairs, indice_pairs, images)
    return delta_array

def calculzate_flux_scales(frames, filt, apass_dir='/Volumes/mice/apass/'):
    """ Calculate flux scales to standize zeropoints """
    try:
        from dfreduce.tasks import calculate_zp
    except ModuleNotFoundError:
        print("calculate_zp in dfreduce not avaialble.")

    zps = []
    for frame in frames:
        results = calculate_zp(frame, filt, catalogue_dir=apass_dir) 
        zp = results.ZP_avg
        zps.append(zp)

    flux_scales = 10**((zps - np.median(zps)) / -2.5)
    
    print("Median zeropoint = {:.2f}".format(np.median(zps)))
    print(f"Flux scales = [{', '.join(map(str, np.around(flux_scales,4)))}]")
    
    return zps, flux_scales


def calculate_sky_offsets(bkg_stats_pairs, indice_pairs, images):
    """ Calculate the background offsets using median/mode of the matched background stats """
    
    meds, modes = [], []
    fig, axes = plt.subplots(2,4, figsize=(22,8))
    for k, color in enumerate(['m','b','c','g','y','orange','r','firebrick']):
        ax = axes.ravel()[k]
        bkg_diff = bkg_stats_pairs[k][0][0,:,0] - bkg_stats_pairs[k][0][1,:,0]
        bkg_diff = bkg_diff[~np.isnan(bkg_diff)]
        
        # Histogram
        counts, bins, _ = ax.hist(bkg_diff, bins=30, range=np.quantile(bkg_diff, [0.001, 0.999]), 
                                  facecolor='None', edgecolor=color, lw=2, alpha=0.5)
        
        # median and mode
        mode = bins[np.argmax(counts)]
        med = np.median(bkg_diff)
        meds.append(med)
        modes.append(mode)

        ax.axvline(med, lw=3, color='k', alpha=0.8)
        ax.axvline(mode, lw=3, color=color, alpha=0.8)
        ax.text(0.08, 0.85, 'med=%.2f\nmode=%.2f'%(med, mode), fontsize=12, transform=ax.transAxes)

    plt.tight_layout()
    plt.show()
    
    for stats, txt in zip([meds, modes], ['median', 'mode']):
        
        # align sky offsets
        delta_0 = 0
        delta_4 = delta_0-stats[7] # 0-4
        delta_3 = delta_0-stats[6] # 0-3
        delta_6 = delta_4-stats[1] # 4-6
        delta_2 = delta_4+stats[3] # 4-2
        delta_5 = 0.5*((delta_6 + stats[0]) + (delta_3 - stats[2]))  # 6-5 + 3-5 
        delta_1 = 0.5*((delta_2 + stats[5]) + (delta_3 + stats[4]))  # 2-1 + 3-1 
        delta_array = np.array([delta_0, delta_1, delta_2,delta_3,delta_4,delta_5,delta_6])
    
        print(f"Sky offsets ({txt}) = [{', '.join(map(str, np.around(delta_array,4)))}]")

        for (i, j) in indice_pairs:
            fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,4))
            fig.subplots_adjust(right=0.95)
            res = images[i]-images[j]
            res_corr = res + delta_array[i] - delta_array[j]

            for ax, img in zip([ax1,ax2],[res, res_corr]):
                m = ax.imshow(img, vmin=-20, vmax=20, cmap='RdYlBu_r')
                ax.set_title("mean offset = {:.3f} ({:s})".format(np.nanmedian(img), txt))

            cax = fig.add_axes([0.95, 0.1, 0.02, 0.8])
            fig.colorbar(m, cax=cax)
            plt.subplots_adjust(wspace=0.05)
        
    return delta_array

def calculate_overlap_stats(images, ind_a, ind_b):
    """ Calculate mean and rms in the overlapping regions of image pairs. """
    
    image_a = images[ind_a] 
    image_b = images[ind_b]
    
    overlap = (~np.isnan(image_a)) & (~np.isnan(image_b))
    
    plt.figure(figsize=(4,3))
    plt.imshow(overlap)
    plt.show()

    Y_overlap, X_overlap = np.where(overlap==True)
    
    xx, yy = np.meshgrid(np.arange(X_overlap.min(), X_overlap.max(), 100), np.arange(Y_overlap.min(), Y_overlap.max(), 100))
    landmarks_pix = np.array([xx, yy]).T.reshape(-1,2)
    
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,5))
    ax1.imshow(image_a, vmin=-20, vmax=60, cmap='gray')
    ax2.imshow(image_b, vmin=-20, vmax=60, cmap='gray')
    for ax in (ax1,ax2):
        for lm in landmarks_pix:
            ax.scatter(lm[0], lm[1], s=5, color='lime')
    plt.show()
    
    fig, (ax) = plt.subplots(1,1, figsize=(7,5))
    plt.imshow(image_a-image_b, vmin=-60, vmax=60, cmap='RdYlBu_r')
    plt.colorbar()
    plt.show()
    
    warnings.simplefilter('ignore', AstropyUserWarning)

    bkg_stats_pair = np.zeros((2, len(landmarks_pix), 2))

    for i, image in enumerate([image_a, image_b]):
        mask = np.isnan(image)
        for j, lm in enumerate(landmarks_pix):
            bkg_val, bkg_rms = get_local_sky(lm, image, mask=mask, box_size=[64,64], pad=32)
            bkg_stats_pair[i, j] = bkg_val, bkg_rms
            
    shape_bkg = xx.T.shape
    
    axis_ratio = (xx.max()-xx.min()) / (yy.max()-yy.min())

    if axis_ratio<=1:
        figsize = (12,int(3/axis_ratio))
    else:
        figsize = (int(3*axis_ratio*3),3)
        
    fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=figsize)
    m=ax1.pcolormesh(xx, yy, bkg_stats_pair[0,:,0].reshape(shape_bkg).T, cmap='viridis', vmin=-10, vmax=60)
    ax1.set_title('Left')
    colorbar(m,ax=ax1)
    m=ax2.pcolormesh(xx, yy, bkg_stats_pair[1,:,0].reshape(shape_bkg).T, cmap='viridis', vmin=-10, vmax=60)
    colorbar(m,ax=ax2)
    ax2.set_title('Right')
    m=ax3.pcolormesh(xx, yy, (bkg_stats_pair[0,:,0]-bkg_stats_pair[1,:,0]).reshape(shape_bkg).T, cmap='RdYlBu_r', vmin=-30, vmax=30)
    colorbar(m,ax=ax3)
    ax3.set_title('Left - Right')
    plt.tight_layout()
    plt.show()
    
    return bkg_stats_pair, landmarks_pix
