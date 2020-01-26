import numpy as np
from classy import Class
from scipy.interpolate import interp1d

class Cosmology:
    """Class to hold the basic cosmology and class attributes.

    This can be initialized by a set of cosmological parameters or a pre-defined name.

    Loaded cosmological models:
    - Quijote: Fiducial cosmology from the Quijote simulations.
    """

    loaded_models = {'Quijote':{"h":0.6711,"omega_cdm":(0.3175 - 0.049)*0.6711**2,
                                "Omega_b":0.049, "sigma8":0.834,"n_s":0.9624,
                                "N_eff":3.04}}

    def __init__(self,redshift,name="",**params):

        """Initialize the cosmology class with cosmological parameters or a defined model.

        Parameters:
        - redshift: Desired redshift
        - name: Load cosmology from a list of predetermined cosmologies. Currently implemented: Quijote
        - **params: Other parameters from CLASS.
        """

        print('need to specify class attributes + methods in the docstring...')

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

    def linear_power(self,kh_vector,kh_min=0.):
        """Compute the linear power spectrum from CLASS for a vector of input k.

        If set, we remove any modes below some k_min.

        Parameters:
        - kh_vector: Vector of wavenumbers (in h/Mpc units) to compute linear power with.
        - kh_min: Value of k (in h/Mpc units) below which to set P(k) = 0.
        """
        print("don't recompute this?")

        # Define output vector and filter modes with too-small k
        output = np.zeros_like(kh_vector)
        filt = np.where(kh_vector>kh_min)
        N_k = len(filt[0])

        # Compute Pk using CLASS (vectorized)
        if not hasattr(self,'vector_linear_power'):
            self.vector_linear_power = np.vectorize(lambda kh: self.cosmo.pk_lin(kh,self.z))

        output[filt] = self.vector_linear_power(kh_vector[filt]*self.h)*self.h**3.
        return output

    def sigmaM(self,M_phys):
        """Compute \sigma(M,z) from CLASS as a vector function.

        Parameters:
        - M_phys: Physical mass in M_sun.
        - z: Redshift.
        """
        # convert to Lagrangian radius
        r_phys = np.power((3.*M_phys)/(4.*np.pi*self.rhoM),1./3.)
        sigma_func = self.vector_sigma_R(r_phys)
        return sigma_func

    def _interpolate_sigma_and_deriv(self,logM_min=6,logM_max=17,npoints=int(1e5)):
        """Create an interpolator function for d ln(sigma)/dlog10(M) and sigma(logM).
        Note that mass is in physical units (without 1/h factor).

        NB: This has no effect if the interpolator has already been computed.

        Parameters:
        - logM_min: Minimum mass in log10(M/Msun)
        - logM_max: Maximum mass in log10(M/Msun)
        - npoints: Number of sampling points.
        """

        if not hasattr(self,'sigma_logM_int'):
            print("Creating an interpolator for sigma(M) and its derivative.")
            ## Compute log derivative by interpolation and numerical differentiation
            # First compute the grid of M and sigma
            M_grid = np.logspace(6,17,10000)
            all_sigM = self.sigmaM(M_grid)
            logM_grid = np.log10(M_grid)

            # Define ln(sigma) and numerical derivatives
            all_lns = np.log(all_sigM)
            all_diff = -np.diff(all_lns)/np.diff(logM_grid)
            mid_logM = 0.5*(logM_grid[:-1]+logM_grid[1:])

            self.sigma_logM_int = interp1d(logM_grid,all_sigM)
            self.dlns_dlogM_int = interp1d(mid_logM,all_diff)

    def _h_over_h0(self):
        """Return the value of H(z)/H(0) at the class redshift"""
        Ea = np.sqrt((self.cosmo.Omega0_cdm()+self.cosmo.Omega_b()+self.cosmo.Omega_Lambda()*pow(self.a,-3)+self.cosmo.Omega0_k()*self.a)/pow(self.a,3))
        return Ea

    def _Omega_m(self):
        """Return the value of Omega_m(z) at the class redshift"""
        hnorm = self._h_over_h0()
        output = (self.cosmo.Omega0_cdm()+self.cosmo.Omega_b())/self.a**3/hnorm**2
        return output
