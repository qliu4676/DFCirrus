import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip

from ..io import logger

try:
    from elderflower.plotting import AsinhNorm, LogNorm, colorbar
except:
    logger.error('elderflower not installed. Missing some utilities.')

def display(image, mask=None,
            k_std=10, cmap="gray_r",
            a=0.1, fig=None, ax=None):
    """ Visualize an image """
    
    if mask is not None:
        sky_vals = image[(mask==0)].ravel()
    else:
        sky_vals = sigma_clip(image, 3).compressed()
        
    sky_mean, sky_std = np.median(sky_vals), np.std(sky_vals)
    
    if fig is None: fig = plt.figure(figsize=(12,8))
    if ax is None: ax = plt.subplot(111)
    ax.imshow(image, cmap="gray_r",
              norm=AsinhNorm(a, vmin=sky_mean-2*sky_std,
                                vmax=sky_mean+k_std*sky_std))
    ax.axis('off')
    return ax
    
def draw_reference_residual(Image):
    """ Plotting function for visualizing output """
    
    mask = Image.mask
    norm = Image.norm
    norm0 = Image.norm0
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(26, 6))
    ax1.imshow(Image.image_ref, norm=norm)
    ax1.set_title('Reference Image')

    ax2.imshow(Image.image_extend_conv, norm=norm)
    ax2.set_title('smoothed')

    ax3.imshow(Image.image_extend_ds, norm=norm)
    ax3.set_title('downsampled')

    ax4.imshow(Image.image_extend_us, norm=norm)
    ax4.set_title('upsampled')
    
    plt.tight_layout()
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    img_ref_res_ = Image.img_ref_res.copy()
    img_ref_res_[mask] = np.nan
    m = ax1.imshow(img_ref_res_, norm=norm)
    colorbar(m, ax = ax1)
    ax1.set_title('Reference Residual')

    img_out_ = Image.image_output.copy()
    img_out_[mask] = np.nan
    m = ax2.imshow(img_out_, norm=norm0)
    colorbar(m, ax = ax2)
    ax2.set_title('Output Residual')
    
    plt.tight_layout()
    plt.show()
