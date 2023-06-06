import numpy as np
from astropy import wcs


### Resampling functions ###

def transform_rescale(val, scale=0.5):
    """ transform coordinates after resampling """
    return (val-1) * scale + scale/2. + 0.5
    
def downsample_wcs(wcs_input, scale=0.5):
    """ Downsample the input wcs along an axis using {CDELT, CRPIX} FITS convention """

    header = wcs_input.to_header(relax=True)
    shape = wcs_input.pixel_shape

    if 'PC1_1' in header.keys():
        cdname = 'PC'
    elif 'CD1_1' in header.keys():
        cdname = 'CD'
    elif 'CDELT1' in header.keys():
        cdname = 'CDELT'
    else:
        msg = 'Fits header has no proper coordinate info (CD, PC, CDELT)!'
        raise KeyError(msg)

    for axis in [1, 2]:
        if cdname == 'PC':
            cd = 'PC{0:d}_{0:d}'.format(axis)
        elif cdname == 'CD':
            cd = 'CD{0:d}_{0:d}'.format(axis)
        elif cdname=='CDELT':
            cd = 'CDELT{0:d}'.format(axis)
            
        cp = 'CRPIX{0:d}'.format(axis)
        na = 'NAXIS{0:d}'.format(axis)

        header[cp] = transform_rescale(header[cp], scale)
        header[cd] = header[cd]/scale
        header[na] = int(round(shape[axis-1]*scale))

    return wcs.WCS(header)
