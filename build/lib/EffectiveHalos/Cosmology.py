import numpy as np
from classy import Class
from scipy.interpolate import interp1d

class Cosmology(object):
    """
    Class to hold the basic cosmology and CLASS attributes. This can be initialized by a set of cosmological parameters or a pre-defined cosmology.

    Loaded cosmological models:

    - **Planck18**: Bestfit cosmology from Planck 2018, using the baseline TT,TE,EE+lowE+lensing likelihood.
    - **Quijote**: Fiducial cosmology from the Quijote simulations of Francisco Villaescusa-Navarro et al.
    - **Abacus**: Fiducial cosmology from the Abacus simulations of Lehman Garrison et al.

    Args:
        redshift (float): Desired redshift :math:`z`
        name (str): Load cosmology from a list of predetermined cosmologies (see above).
        params (kwargs): Any other CLASS parameters.

    Keyword Args:
        verb (bool): If true output useful messages througout run-time, default: False.

    """

    loaded_models = {'Quijote':{"h":0.6711,"omega_cdm":(0.3175 - 0.049)*0.6711**2,
                                "Omega_b":0.049, "sigma8":0.834,"n_s":0.9624,
                                "N_eff":3.04},
                     'Abacus':{"h":0.6726,"omega_cdm":0.1199,
                                "omega_b":0.02222,"n_s":0.9652,"sigma8":0.830,
                                "N_eff":3.04},
                     'Planck18':{"h":0.6732,"omega_cdm":0.12011,"omega_b":0.022383,
                                "n_s":0.96605,"sigma8":0.8120}}

    def __init__(self,redshift,name="",verb=False,**params):

        """
        Initialize the cosmology class with cosmological parameters or a defined model.

        """
        ## Load parameters into a dictionary to pass to CLASS
        class_params = dict(**params)
        if len(name)>0:
            if len(params.items())>0:
                raise Exception('Must either choose a preset cosmology or specify parameters!')
            if name in self.loaded_models.keys():
                if verb: print('Loading the %s cosmology at z = %.2f'%(name,redshift))
                loaded_model = self.loaded_models[name]
                for key in loaded_model.keys():
                    class_params[key] = loaded_model[key]
            else:
                raise Exception("This cosmology isn't yet implemented")
        else:
            if len(params.items())==0:
                if verb: print('Using default CLASS cosmology')
            for name, param in params.items():
                class_params[name] = param

        ## # Check we have the correct parameters
        if 'sigma8' in class_params.keys() and 'A_s' in class_params.keys():
            raise NameError('Cannot specify both A_s and sigma8!')

        ## Define other parameters
        self.z = redshift
        self.a = 1./(1.+redshift)
        if 'output' not in class_params.keys():
            class_params['output']='mPk'
        if 'P_k_max_h/Mpc' not in class_params.keys() and 'P_k_max_1/Mpc' not in class_params.keys():
            class_params['P_k_max_h/Mpc']=100.
        if 'z_pk' in class_params.keys():
            assert class_params['z_pk']==redshift, "Can't pass multiple redshifts!"
        else:
            class_params['z_pk']=redshift

        ## Load CLASS and set parameters
        if verb: print('Loading CLASS')
        self.cosmo = Class()
        self.cosmo.set(class_params)
        self.cosmo.compute()
        self.h = self.cosmo.h()
        self.name = name
        self.verb = verb

        ## Create a vectorized sigma(R) function from CLASS
        self.vector_sigma_R = np.vectorize(lambda r: self.cosmo.sigma(r/self.h,self.z))

        # get density in physical units at z = 0
        # rho_critical is in Msun/h / (Mpc/h)^3 units
        # rhoM is in **physical** units of Msun/Mpc^3
        self.rho_critical = ((3.*100.*100.)/(8.*np.pi*6.67408e-11)) * (1000.*1000.*3.085677581491367399198952281E+22/1.9884754153381438E+30)
        self.rhoM = self.rho_critical*self.cosmo.Omega0_m()

    def compute_linear_power(self,kh,kh_min=0.):
        """Compute the linear power spectrum from CLASS for a vector of input k.

        If set, we remove any modes below some minimum k.

        Args:
            kh (float, np.ndarray): Wavenumber or vector of wavenumbers (in h/Mpc units) to compute linear power with.
            kh_min (float): Value of k (in h/Mpc units) below which to set :math:`P(k) = 0`, default: 0.

        Returns:
            np.ndarray: Linear power spectrum in :math:`(h^{-1}\mathrm{Mpc})^3` units
        """

        if type(kh)==np.ndarray:

            # Define output vector and filter modes with too-small k
            output = np.zeros_like(kh)
            filt = np.where(kh>kh_min)
            N_k = len(filt[0])

            # Compute Pk using CLASS (vectorized)
            if not hasattr(self,'vector_linear_power'):
                ## NB: This works in physical 1/Mpc units
                self.vector_linear_power = np.vectorize(lambda k: self.cosmo.pk_lin(k*self.h,self.z)*self.h**3.)

            output[filt] = self.vector_linear_power(kh[filt])
            return output

        else:
            if kh<kh_min:
                return 0.
            else:
                return self.vector_linear_power(kh)

    def sigma_logM_int(self,logM_h):
        """Return the value of :math:`\sigma(M,z)` using the prebuilt interpolators, which are constructed if not present.

        Args:
            logM (np.ndarray): Input :math:`\log_{10}(M/h^{-1}M_\mathrm{sun})`

        Returns:
            np.ndarray: :math:`\sigma(M,z)`
        """
        if not hasattr(self,'sigma_logM_int_func'):
            self._interpolate_sigma_and_deriv()
        return self._sigma_logM_int_func(logM_h)

    def dlns_dlogM_int(self,logM_h):
        """Return the value of :math:`d\ln\sigma/d\log M` using the prebuilt interpolators, which are constructed if not present.

        Args:
            logM (np.ndarray): Input :math:`\log_{10}(M/h^{-1}M_\mathrm{sun})`

        Returns:
            np.ndarray: :math:`d\ln\sigma/d\log M`
        """
        if not hasattr(self,'dlns_dlogM_int_func'):
            self._interpolate_sigma_and_deriv()
        return self._dlns_dlogM_int_func(logM_h)

    def _sigmaM(self,M_h):
        """Compute :math:`\sigma(M,z)` from CLASS as a vector function.

        Args:
            M_h (np.ndarray): Mass in :math:`h^{-1}M_\mathrm{sun}` units.
            z (float): Redshift.

        Returns:
            np.ndarray: :math:`\sigma(M,z)`
        """
        # convert to Lagrangian radius
        r_h = np.power((3.*M_h)/(4.*np.pi*self.rhoM),1./3.)
        sigma_func = self.vector_sigma_R(r_h)
        return sigma_func

    def _interpolate_sigma_and_deriv(self,logM_h_min=6,logM_h_max=17,npoints=int(1e5)):
        """Create an interpolator function for :math:`d\ln\sigma/d\log M` and :math:`sigma(M)`.

        NB: This has no effect if the interpolator has already been computed.

        Keyword Args:
            logM_min (float): Minimum mass in :math:`\log_{10}(M/h^{-1}M_\mathrm{sun})`, default: 6
            logM_max (float): Maximum mass in :math:`\log_{10}(M/h^{-1}M_\mathrm{sun})`, default 17
            npoints (int): Number of sampling points, default 100000

        """

        if not hasattr(self,'_sigma_logM_int_func'):
            if self.verb: print("Creating an interpolator for sigma(M) and its derivative.")
            ## Compute log derivative by interpolation and numerical differentiation
            # First compute the grid of M and sigma
            M_h_grid = np.logspace(logM_h_min,logM_h_max,10000)
            all_sigM = self._sigmaM(M_h_grid)
            logM_h_grid = np.log10(M_h_grid)

            # Define ln(sigma) and numerical derivatives
            all_lns = np.log(all_sigM)
            all_diff = -np.diff(all_lns)/np.diff(logM_h_grid)
            mid_logM_h = 0.5*(logM_h_grid[:-1]+logM_h_grid[1:])

            self._sigma_logM_int_func = interp1d(logM_h_grid,all_sigM)
            self._dlns_dlogM_int_func = interp1d(mid_logM_h,all_diff)

    def _h_over_h0(self):
        """Return the value of :math:`H(z)/H(0)` at the class redshift

        Returns:
            float: :math:`H(z)/H(0)`
        """
        Omega0_k = 1.-self.cosmo.Omega0_m()-self.cosmo.Omega_Lambda()
        Ea = np.sqrt((self.cosmo.Omega0_m()+self.cosmo.Omega_Lambda()*pow(self.a,-3)+Omega0_k*self.a)/pow(self.a,3))
        return Ea

    def _Omega_m(self):
        """Return the value of :math:`\Omega_m(z)` at the class redshift

        Returns:
            float: :math:`\Omega_m(z)`
        """
        hnorm = self._h_over_h0()
        output = (self.cosmo.Omega0_m())/self.a**3/hnorm**2
        return output
