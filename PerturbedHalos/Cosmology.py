import numpy as np
from classy import Class
from scipy.interpolate import interp1d

class Cosmology(object):
    """
    Class to hold the basic cosmology and class attributes.

    This can be initialized by a set of cosmological parameters or a pre-defined name.

    Loaded cosmological models:

    - Planck18: Bestfit cosmology from Planck 2018, using the baseline TT,TE,EE+lowE+lensing likelihood.
    - Quijote: Fiducial cosmology from the Quijote simulations of Francisco Villaescusa-Navarro et al.
    - Abacus: Fiducial cosmology from the Abacus simulations of Lehman Garrison et al.

    """

    def __init__(self,redshift,name="",**params):

        """
        Initialize the cosmology class with cosmological parameters or a defined model.

        Args:
            redshift (float): Desired redshift
            name (str): Load cosmology from a list of predetermined cosmologies. Currently implemented: Quijote
            **params (**kwargs): Other parameters from CLASS.
        """
        pass
