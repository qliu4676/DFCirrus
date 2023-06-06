import os
import numpy as np

from astropy.io import fits
from astropy.coordinates import Galactic

from reproject import reproject_from_healpix

class PlanckImage:

    """ A class to read and reproject Planck maps. """
    
    def __init__(self, hdu_path):
    
        if not os.path.exists(hdu_path):
            raise FileNotFoundError(['Planck dust model map not found!'])
    
        with fits.open(hdu_path) as hdul: 
            hdu = hdul[1]
            self.header = hdul[1].header

            self.radiance = hdu.data['RADIANCE']
            self.tau = hdu.data['TAU353']
            self.ebv = hdu.data['EBV']

            hdul.close()
            
        self.data = dict(radiance=self.radiance, tau=self.tau, EBV=self.ebv)
    
    def reproject(self, wcs, shape, model='radiance'):
        """
        Create a map reprojected from Healpix data.

        Parameters
        ----------
        wcs: astropy.wcs.WCS
            Input WCS.
        shape: tuple
            Shape of the output map.
        model: 'radiance' or 'tau' or 'EBV'
            Planck dust model in use.
            
        Returns
        -------
        dust_map_rp:
            Output dust map.
        """
        
        dust_map = self.data.get(model)
        
        if dust_map is not None:
            dust_map_rp, _ = reproject_from_healpix((dust_map, Galactic()), output_projection=wcs, shape_out=shape, nested=True)
            return dust_map_rp
