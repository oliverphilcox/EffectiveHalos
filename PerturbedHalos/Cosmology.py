class Cosmology(object):
    """
    Class to hold the basic cosmology and class attributes.

    This can be initialized by a set of cosmological parameters or a pre-defined name.

    Loaded cosmological models:

    - Planck18: Bestfit cosmology from Planck 2018, using the baseline TT,TE,EE+lowE+lensing likelihood.
    - Quijote: Fiducial cosmology from the Quijote simulations of Francisco Villaescusa-Navarro et al.
    - Abacus: Fiducial cosmology from the Abacus simulations of Lehman Garrison et al.

    """

    loaded_models = {'Quijote':{"h":0.6711,"omega_cdm":(0.3175 - 0.049)*0.6711**2,
                                "Omega_b":0.049, "sigma8":0.834,"n_s":0.9624,
                                "N_eff":3.04},
                     'Abacus':{"h":0.6726,"omega_cdm":0.1199,
                                "omega_b":0.02222,"n_s":0.9652,"sigma8":0.830,
                                "N_eff":3.04},
                     'Planck18':{"h":0.6732,"omega_cdm":0.12011,"omega_b":0.022383,
                                "n_s":0.96605,"sigma8":0.8120}}

    def __init__(self,redshift,name="",**params):

        """
        Initialize the cosmology class with cosmological parameters or a defined model.

        Args:
            redshift (float): Desired redshift
            name (str): Load cosmology from a list of predetermined cosmologies. Currently implemented: Quijote
            **params (**kwargs): Other parameters from CLASS.
        """
        return 0
