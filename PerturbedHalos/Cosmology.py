class Cosmology(object):
    """
    Class to hold the basic cosmology and class attributes.

    This can be initialized by a set of cosmological parameters or a pre-defined name.

    Loaded cosmological models:

    - **Planck18**: Bestfit cosmology from Planck 2018, using the baseline TT,TE,EE+lowE+lensing likelihood.
    - **Quijote**: Fiducial cosmology from the Quijote simulations of Francisco Villaescusa-Navarro et al.
    - **Abacus**: Fiducial cosmology from the Abacus simulations of Lehman Garrison et al.

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
        ## Load parameters into a dictionary to pass to CLASS
        class_params = dict(**params)
        if len(name)>0:
            if len(params.items())>0:
                raise Exception('Must either choose a preset cosmology or specify parameters!')
            if name in self.loaded_models.keys():
                print('Loading the %s cosmology at z = %.2f'%(name,redshift))
                loaded_model = self.loaded_models[name]
                for key in loaded_model.keys():
                    class_params[key] = loaded_model[key]
            else:
                raise Exception("This cosmology isn't yet implemented")
        else:
            if len(params.items())==0:
                print('Using default CLASS cosmology')
            for name, param in params.items():
                class_params[name] = param

        ## # Check we have the correct parameters
        if 'sigma8' in class_params.keys() and 'A_s' in class_params.keys():
            raise NameError('Cannot specify both A_s and sigma8!')

        ## Define other parameters
        self.z = redshift
        self.a = 1./(1.+redshift)
        print('Add other CLASS parameters here?')
        if 'output' not in class_params.keys():
            class_params['output']='mPk'
        if 'P_k_max_h/Mpc' not in class_params.keys() and 'P_k_max_1/Mpc' not in class_params.keys():
            class_params['P_k_max_h/Mpc']=100.
        if 'z_pk' in class_params.keys():
            assert class_params['z_pk']==redshift, "Can't pass multiple redshifts!"
        else:
            class_params['z_pk']=redshift

        ## Load CLASS and set parameters
        print('Loading CLASS')
        self.cosmo = Class()
        self.cosmo.set(class_params)
        self.cosmo.compute()
        self.h = self.cosmo.h()

        ## Create a vectorized sigma(R) function from CLASS
        self.vector_sigma_R = np.vectorize(lambda r: self.cosmo.sigma(r,self.z))

        # get density in physical units at z = 0
        self.rho_critical = ((3.*100.*100.)/(8.*np.pi*6.67408e-11)) * (1000.*1000.*3.085677581491367399198952281E+22/1.9884754153381438E+30)
        self.rhoM = self.rho_critical*self.cosmo.Omega0_m()*self.cosmo.h()**2.
