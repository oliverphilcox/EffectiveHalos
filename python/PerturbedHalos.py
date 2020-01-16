from classy import Class
from scipy.interpolate import interp1d
import numpy as np
import sys
sys.path.append('/home/ophilcox/FAST-PT/')
import FASTPT as FASTPT

class Cosmology:
    """Class to hold the basic cosmology and class attributes.

    This can be initialized by a set of cosmological parameters or a pre-defined name.

    Loaded cosmological models:
    - Quijote: Fiducial cosmology from the Quijote simulations.
    """

    loaded_models = {'Quijote':{"h":0.6711,"Omega_cdm":(0.3175 - 0.049)*0.6711**2,
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

        ## Load parameters
        if len(name)>0:
            if name in self.loaded_models.keys():
                print('Loading the %s cosmology at z = %.2f'%(name,redshift))
                loaded_model = self.loaded_models[name]
                for key in loaded_model.keys():
                    setattr(self,key,loaded_model[key])
            else:
                raise Exception("This cosmology isn't yet implemented")
        else:
            if len(params.items())==0:
                print('Using default CLASS cosmology')
            for name, param in params.items():
                setattr(self,name,param)

        ## # Check we have the correct parameters
        if hasattr(self,'sigma8') and hasattr(self,'A_s'):
            raise NameError('Cannot specify both A_s and sigma8!')

        ## Define other parameters
        self.z = redshift
        self.a = 1./(1.+redshift)
        print('Add other CLASS parameters here?')
        class_params = dict(**params)
        if 'output' not in class_params.keys():
            class_params['output']='mPk'
        if 'P_k_max_h/Mpc' not in class_params.keys() and 'P_k_max_1/Mpc' not in class_params.keys():
            class_params['P_k_max_h/Mpc']=10.
        if 'z_pk' in class_params.keys():
            assert class_params['z_pk']==redshift, "Can't pass multiple redshifts!"
        else:
            class_params['z_pk']=redshift

        ## Load CLASS and set parameters
        self.cosmo = Class()
        self.cosmo.set(class_params)
        self.cosmo.compute()

        ## Create a vectorized sigma(R) function from CLASS
        self.vector_sigma_R = np.vectorize(self.cosmo.sigma)

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

        # Compute Pk using fast function from CLASS
        output[filt] = self.cosmo.get_pk_array(kh_vector[filt],np.atleast_1d(self.z),len(filt[0]),1,0)

        return output

    def sigmaM(self,M_phys):
        """Compute \sigma(M,z) from CLASS as a vector function.

        Parameters:
        - M_phys: Physical mass in M_sun.
        - z: Redshift.
        """
        # convert to Lagrangian radius
        r_phys = np.power((3.*M_phys)/(4.*np.pi*self.rhoM),1./3.)
        sigma_func = self.vector_sigma_R(r_phys,self.z)
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

            print("Should this be dln/dlog??")

    def _h_over_h0(self):
        """Return the value of H(z)/H(0) at the class redshift"""
        Ea = np.sqrt((self.cosmo.Omega0_cdm()+self.cosmo.Omega_b()+self.cosmo.Omega_Lambda()*pow(self.a,-3)+self.cosmo.Omega0_k()*self.a)/pow(self.a,3))
        return Ea

    def _Omega_m(self):
        """Return the value of Omega_m(z) at the class redshift"""
        hnorm = self._h_over_h0()
        output = (self.cosmo.Omega0_cdm()+self.cosmo.Omega_b())/self.a**3/hnorm**2
        return output


class MassFunction:
    """Class to hold a mass function for halos and associated bias.

    Implemented Mass Functions:
    - 'Sheth-Tormen': Sheth-Tormen 1999 analytic mass function. This assumes virialized halos.
    - 'Tinker': Tinker et al. 2010, eq. 8. This assume a spherical overdensity, the value of which can be specified.
    - 'Crocce': Crocce et al. 2009, eq. 22. Calibrated from Friends-of-Friends halos with linking length 0.2

    Implemented Bias Functions:
    - 'Sheth-Tormen': Sheth-Tormen 2001 ???. Associated to the 'Sheth-Tormen' mass function.
    - 'Tinker': Tinker et al. 2010 ???. Associated to the 'Tinker' mass function.
    - 'Crocce': ???. Associated to the 'Crocce' mass function.
    """

    def __init__(self,cosmology,mass_function_name='Crocce',**mass_params):
        """Initialize the class by loading the relevant model parameters.

        Parameters:
        - cosmology: Instance of the Cosmology class containing relevant cosmological information
        - mass_function_name: Mass function to use (see above), default: 'Crocce'
        - mass_params: Any additional parameters to pass to the class. These include:
            - tinker_overdensity: (Only for the Tinker mass function): spherical overdensity defining halos, default: 200
        """

        print('need to specify class attributes + methods in the docstring...')

        print('add mass function + bias references')
        print('add ref to ST overdensity (NakamuraSuto)')

        # Write attributes, if they're of the correct type
        self.mass_function_name = mass_function_name
        if isinstance(cosmology,Cosmology):
            self.cosmology = cosmology
        else:
            raise NameError('cosmology attribute must be an instance of the Cosmology class!')

        # Expansion parameter
        self.a = 1./(1.+self.cosmology.z)

        # reduced H0
        self.h = self.cosmology.cosmo.h()

        # Set hyperparameters to default if not specified
        mass_dict = dict(**mass_params)
        if mass_function_name=='Tinker':
            if 'tinker_overdensity' not in mass_dict.keys():
                mass_dict['tinker_overdensity']=200.

        # Define mass-function specific parameters:
        self._load_mass_function_parameters(mass_dict)


    def mass_function(self,m_h):
        """We currently assume that all mass functions can be written as
        dn/dlogM = f(sigma_M) * rho_matter * d(log sigma_M)/d(log10 M) / M
        where sigma_M^2 is the overdensity variance on spheres with a
        radius given by the Lagrangian radius for mass M.

        For details of the available parametrizations, see the class description.

        Parameters:
        - m_h: Mass in Msun/h units.
        """
        m = m_h/self.h # mass in Msun units

        logM = np.log10(m) # log10(M/Msun)
        dlns_dlogM = self.cosmology.dlns_dlogM_int(logM)

        print('combine with halo bias?')
        # Compute peak height
        sigma = self.cosmology.sigma_logM_int(logM)
        nu = self.delta_c/sigma

        # Compute universal mass function f(nu)
        if self.mass_function_name=='Sheth-Tormen':
            f = nu * self.A_ST * (1. + (self.a_ST * nu**2)**(-self.p_ST)) * np.exp(-self.a_ST * nu**2/2.)
        elif self.mass_function_name=='Tinker':
            f = self.alpha*(1.+np.power(self.beta*nu,-2.*self.phi))*pow(nu,2.*self.eta)*np.exp(-self.gamma*nu**2/2.)
        elif self.mass_function_name=='Crocce':
            f = self.pA*(np.power(sigma,-self.pa)+self.pb)*np.exp(-self.pc/sigma**2)

        # Return mass function
        mf = f * self.cosmology.rhoM * dlns_dlogM / m
        return mf


    def linear_halo_bias(self,m_h):
        """Compute the linear halo bias, from the peak background split.
        Associated bias functions are available for each mass function, and more can be user-defined.
        See the class description for details of the loaded parametrizations.

        Parameters:
        - m_h: Mass in Msun/h units.
        """

        ## From Sheth-Tormen mass function in Peak-Background split
        m = m_h/self.h

        logM = np.log10(m)
        sigma= self.cosmology.sigma_logM_int(logM);
        nu = self.delta_c/sigma;

        if self.mass_function_name=='Sheth-Tormen':
            return 1.+(self.a_ST*np.power(nu,2)-1.+2.*self.p_ST/(1.+np.power(self.a_ST*np.power(nu,2),self.p_ST)))/self.delta_c;

        elif self.mass_function_name=='Tinker':
            return 1.-self.fit_A*np.power(nu,self.fit_a)/(np.power(nu,self.fit_a)+np.power(self.delta_c,self.fit_a))+self.fit_B*np.power(nu,self.fit_b)+self.fit_C*np.power(nu,self.fit_c);

        elif self.mass_function_name=='Crocce':
            return 1.-self.pa/(self.delta_c + self.pb*self.delta_c*(self.delta_c/nu)**self.pa) + (2.*self.pc*np.power(nu,2.))/np.power(self.delta_c,3.)



    def _load_mass_function_parameters(self,mass_dict):
        """Load internal mass function parameters.

        Parameters:
        - mass_function: Mass function name.
        - mass_dict: Dictionary of additional parameters."""

        if self.mass_function_name=='Sheth-Tormen':

            # Load mass function parameters and normalization
            from scipy.special import gamma as gamma_fn
            self.p_ST = 0.3
            self.a_ST = 0.707
            self.A_ST  = (1.+2.**(-self.p_ST)*gamma_fn(0.5-self.p_ST)/np.sqrt(np.pi))**(-1.)*np.sqrt(2.*self.a_ST/np.pi)

            # Compute the spherical collapse threshold of Nakamura-Suto.
            Om_mz = self.cosmology._Omega_m()
            dc0 = (3./20.)*pow(12.*np.pi,2./3.);
            self.delta_c = dc0*(1.+0.012299*np.log10(Om_mz));

        elif self.mass_function_name=='Tinker':
            self.delta_c = 1.686 # critical density for collapse

            # MASS FUNCTION PARAMETERS
            ## Compute model parameters from interpolation given odelta value
            odelta = mass_dict['tinker_overdensity']
            alpha=np.asarray([0.368,0.363,0.385])
            beta0=np.asarray([0.589,0.585,0.544])
            gamma0=np.asarray([0.864,0.922,0.987])
            phi0=np.asarray([-0.729,-0.789,-0.910])
            eta0=np.asarray([-0.243,-0.261,-0.261])
            odeltas=np.asarray([200,300,400])
            alpha=interp1d(odeltas,alpha)(odelta)
            beta0=interp1d(odeltas,beta0)(odelta)
            gamma0=interp1d(odeltas,gamma0)(odelta)
            phi0=interp1d(odeltas,phi0)(odelta)
            eta0=interp1d(odeltas,eta0)(odelta)

            self.beta = beta0*self.a**-0.2
            self.phi = phi0*self.a**0.08
            self.eta = eta0*self.a**-0.27
            self.gamma = gamma0*self.a**0.01

            # BIAS FUNCTION PARAMETERS
            y = log10(odelta);

            self.fit_A = 1.0 + 0.24*y*exp(-pow(4./y,4.));
            self.fit_a = 0.44*y-0.88;
            self.fit_B = 0.183;
            self.fit_b = 1.5;
            self.fit_C = 0.019+0.107*y+0.19*exp(-pow(4./y,4.));
            self.fit_c = 2.4;

        elif self.mass_function_name=='Crocce':
            self.delta_c = 1.686  # critical density for collapse

            self.pA = 0.58*self.a**0.13
            self.pa = 1.37*self.a**0.15
            self.pb = 0.3*self.a**0.084
            self.pc = 1.036*self.a**0.024

        else:
            raise NameError('Mass function %s not currently implemented!'%self.mass_function_name)

class HaloPhysics:
    """Class to hold halo model quantities and relevant integrals.

    Implemented Concentration Functions:
    - 'Duffy': Duffy et al. (2008) for virial-density haloes (second section in Table 1)

    Implemented Halo Profiles:
    - 'NFW':
    """

    def __init__(self,cosmology,mass_function,concentration_name='Duffy',profile_name='NFW',**hyperparams):
        """Initialize the class with relevant model hyperparameters.

        Parameters:
        - cosmology: Instance of the Cosmology class containing relevant cosmology and functions.
        - mass_function: Instance of the MassFunction class, containing the mass function and bias
        - concentration_name: Concentration parametrization to use (see above), default: 'Duffy'
        - profile_name: Halo profile parametrization to use (see above), default: 'NFW'
        - hyperparams: Any additional parameters to pass to the class. These include:
            - logM_min: Minimum mass in log10(M/Msun), default: 6
            - logM_max: Maximum mass in log10(M/Msun), default: 17
            - npoints: Number of sampling points for sigma(M) interpolation, default: 1e5
            - tinker_overdensity: (Only for the Tinker mass function): spherical overdensity defining halos, default: 200
        """
        print('Add detail on mass functions from CCL.')
        print('Add concentration details + use this argument')
        print('Add profile details and use this argument')
        print('Add NFW reference')
        print('Are these hyperparams overkill?')


        print('need to specify class attributes + methods in the docstring...')

        # Write attributes, if they're of the correct type
        if isinstance(cosmology, Cosmology):
            self.cosmology = cosmology
        else:
            raise TypeError('cosmology input must be an instance of the Cosmology class!')
        if isinstance(mass_function, MassFunction):
            self.mass_function = mass_function
        else:
            raise TypeError('mass_function input must be an instance of the MassFunction class!')
        self.concentration_name = concentration_name
        self.profile_name = profile_name

        # Set hyperparameters to default if not specified
        self.hyper_dict = dict(**hyperparams)
        if 'logM_min' not in self.hyper_dict.keys():
            self.hyper_dict['logM_min']=6.
        if 'logM_max' not in self.hyper_dict.keys():
            self.hyper_dict['logM_max']=17.
        if 'npoints' not in self.hyper_dict.keys():
            self.hyper_dict['npoints']=int(1e5)

        # Create interpolators for sigma and d(ln(sigma))/dlog10(M):
        cosmology._interpolate_sigma_and_deriv(self.hyper_dict['logM_min'],self.hyper_dict['logM_max'],self.hyper_dict['npoints'])

        # Save reduced Hubble value for later use
        self.h = self.cosmology.cosmo.h()
        self.a = self.cosmology.a

    def halo_profile(self,m_h,k_phys):
        """Compute the halo profile function in Fourier space; rho(k|m) = m/rhoM*u(k|m)
        where rhoM is the background matter density and u(k|m) is the halo profile.

        We assume halos have a virial collapse overdensity here.

        For details of the available profile parametrizations, see the class description.

        Parameters:
        - m_h: Mass in Msun/h units.
        - k_phys: Wavenumber in 1/Mpc units.
        """

        m = m_h/self.h # in Msun units

        # Compute virial overdensity
        print('should this be a virial overdensity?')
        odelta = self._virial_overdensity()

        # The halo virial radius in physical units
        rv = np.power(m*3.0/(4.0*np.pi*self.cosmology.rhoM*odelta),1.0/3.0)

        # Compute halo concentration
        c = self._halo_concentration(m_h);
        # The function u is normalised to 1 for k<<1 so multiplying by M/rho turns units to a density
        return self._normalized_halo_profile(k_phys,rv, c)*m/self.cosmology.rhoM;

    def _virial_overdensity(self):
        """Compute the virial collapse overdensity from Bryan-Norman 1998"""
        Om_mz = self.cosmology._Omega_m()
        x = Om_mz-1.;
        Dv0 = 18.*pow(np.pi,2);
        Dv = (Dv0+82.*x-39.*pow(x,2))/Om_mz;

        return Dv;

    def _halo_concentration(self,m_h):
        """Compute the halo concentration c = r_virial / r_scale.

        For details of the available concentration parametrizations, see the class description.

        Parameters:
        - m_h: Mass in Msun/h units.
        """

        m = m_h/self.h

        if self.concentration_name=='Duffy':
            m_pivot = 2e12/self.h;
            return 7.85*np.power(m/m_pivot,-0.081)*pow(self.a,0.71);
        else:
            raise NameError('Concentration profile %s is not implemented yet'%(self.concentration_name))


    def _normalized_halo_profile(self,k_phys,r_virial,c):
        """Compute the normalized halo profile function in Fourier space; u(k|m)

        For details of the available profile parametrizations, see the class description.

        Note that the function returns unity for k_phys < = 0.

        Parameters:
        - k_phys: Wavenumber in 1/Mpc units.
        - r_virial: Virial radius in Mpc units.
        - c: Halo concentration parameter; c = r_virial/r_scale.
        """

        if self.profile_name=='NFW':
            from scipy.special import sici

            r_scale = r_virial/c

            # Check if we have an array or a float input and compute accordingly
            if type(k_phys)==np.ndarray:
                # filter out sections with k_phys<=0
                filt = np.where(k_phys>0)

                # compute the matrix of r_scale * k_phys
                ks0 = np.matmul(k_phys.reshape(-1,1),r_scale.reshape(1,-1))
                ks = ks0[filt,:]
            else:
                if k_phys<=0.:
                    return 1.
                ks = k_phys*r_scale

            f1 = np.sin(ks)*(sici(ks*(1.+c))[0]-sici(ks)[0]);
            f2 = cos(ks)*(sici(ks*(1.+c))[1]-sici(ks)[1]);
            f3 = np.sin(c*ks)/(ks*(1.+c));
            fc = np.log(1.+c)-c/(1.+c);

            if type(k_phys)==np.ndarray:
                output = np.ones_like(ks0)
                output[filt,:]=((f1+f2-f3)/fc)
                return output
            else:
                return (f1+f2-f3)/fc
        else:
            raise NameError('Halo profile %s is not implemented yet'%(self.profile_name))

class MassIntegrals:
    """Class to compute and store the various mass integrals of the form
        $$I_p^{q1,q2}(k_1,...k_p) = \int n(m)b^{(q1)}(m)b^{(q2)}\frac{m^p}{\rho^p}u(k_1|m)..u(k_p|m)dm$$

    which are needed to compute the power spectrum model. Here b^{(q)} is the q-th order bias (with b^{0}=1),
    u(k|m) is the normalized halo profile and n(m) is the mass function.

    All integrals are performed via Simpson's rule over a specified mass range.

    For the I_1^{1,0} = I_1^1 integral, the integral must be corrected to ensure that we recover the bias consistency relation;
        $ I_1^1 \rightarrow 1 $ as $ k \rightarrow 0 $.
    This requires an infinite mass range, so we instead approximate;
        $ I_1^1(k)_\mathrm{corrected} = I_1^1(k) + ( 1 - I_1^1(0) ) u(k|m_min) / u(k|0) $
    for normalized halo profile $u$.
    """
    def __init__(self,cosmology,mass_function,halo_physics,kh_vector,min_logM=6.001, max_logM=16.999, N_mass=int(1e4)):
        """Initialize the class with relevant model hyperparameters.

        Parameters:
        - cosmology: Instance of the Cosmology class containing relevant cosmology and functions.
        - mass_function: Instance of the MassFunction class, containing the mass function and bias.
        - halo_physics: Instance of the HaloPhysics class, containing the halo profiles and concentrations.
        - kh_vector: Array (or float) of wavenumbers in h/Mpc units from which to compute mass integrals.
        - min_logM: Minimum mass in log10(M/Msun) units, default: 6.001.
        - max_logM: Maximum mass in log10(M/Msun) units, default: 16.999.
        - N_mass: Number of logarithmically spaced mass grid points, default: 10000.
        """

        print('need to specify class attributes + methods in the docstring...')
        print('sensibly set default n_mass')

        # Write attributes, if they're of the correct type
        if isinstance(cosmology, Cosmology):
            self.cosmology = cosmology
        else:
            raise TypeError('cosmology input must be an instance of the Cosmology class!')
        if isinstance(mass_function, MassFunction):
            self.mass_function = mass_function
        else:
            raise TypeError('mass_function input must be an instance of the MassFunction class!')
        if isinstance(halo_physics, HaloPhysics):
            self.halo_physics = halo_physics
        else:
            raise TypeError('halo_physics input must be an instance of the HaloPhysics class!')

        print('we should shunt a lot of these definitions into the NonLinearPower class? or Cov class?')

        # Run some important checks
        print('find better place to store these parameters')
        interp_min = self.halo_physics.hyper_dict['logM_min']
        interp_max = self.halo_physics.hyper_dict['logM_max']
        assert min_logM >= interp_min, 'Minimum mass must be greater than the interpolation limit (10^%.2f)'%interp_min
        assert max_logM <= interp_max, 'Minimum mass must be less than the interpolation limit (10^%.2f)'%interp_max

        # Load reduced H0 for clarity
        self.h = self.cosmology.cosmo.h()

        # Save other attributes
        self.min_logM = min_logM
        self.max_logM = max_logM
        self.N_mass = N_mass

        # Define a mass vector for computations
        self.logM_grid = np.linspace(self.min_logM,self.max_logM, self.N_mass)

        # Load and convert masses and wavenumbers
        m = np.power(10.,self.logM_grid)
        self.m_h_grid = m*self.h # in Msun/h

        self.k_vectors = kh_vector*self.h # remove h dependence so that k is in physical units
        self.N_k = len(self.k_vectors) # define number of k points

    def compute_I_11(self,apply_correction = True):
        """Compute the I_1^1(k) integral, if not already computed. Also apply the correction noted in the class header.

        When computing {}_i J_1^1 type integrals (over a finite mass bin), the correction should *not* be applied.

        Parameters:
        - apply_correction (Boolean): Whether to apply the correction in the class header to ensure the bias consistency relation is upheld."""

        if not hasattr(self,'I_11'):
            # Compute the integral over the utilized mass range
            self.I_11 = simps(self._I_p_q1q2_integrand(1,1,0,self.k_vectors),self.logM_grid,axis=1)

            if apply_correction:
                A = 1. - simps(self._I_p_q1q2_integrand(1,1,0,0.),self.logM_grid)
                # compute window functions
                min_m_h = np.power(10.,self.min_logM)*self.h
                min_window = self.halo_physics.halo_profile(min_m_h,self.k_vectors).ravel()
                zero_window = self.halo_physics.halo_profile(min_m_h,0.).ravel()
                self.I_11 += A*min_window/zero_window
        return self.I_11

    def compute_I_20(self):
        """Compute the I_2^0(k,k) integral, if not already computed.
        Note that we assume both k vectors are the same here."""
        if not hasattr(self,'I_20'):
            k_stack = np.vstack([self.k_vectors,self.k_vectors])
            self.I_20 = simps(self._I_p_q1q2_integrand(2,0,0,k_stack),self.logM_grid,axis=1)
            print('How do we want this - should the two k arguments be the same?')
            print('cleanup this')
            print('save the k stack?')
        return self.I_20

    def compute_I_21(self):
        """Compute the I_2^1 integral, if not already computed.
        Note that we assume both k vectors are the same here."""
        if not hasattr(self,'I_21'):
            k_stack = np.vstack([self.k_vectors,self.k_vectors])
            self.I_21 = simps(self._I_p_q1q2_integrand(2,1,0,k_stack),self.logM_grid,axis=1)
            print('How do we want this - should the two k arguments be the same?')
            print('cleanup this')
            print('save the k stack?')
        return self.I_21

    def compute_I_111(self):
        """Compute the I_1^{1,1} integral, if not already computed.
        """
        if not hasattr(self,'I_111'):
            self.I_111 = simps(self._I_p_q1q2_integrand(1,1,1,self.k_vectors),self.logM_grid,axis=1)
        return self.I_111

    def compute_I_01(self):
        """Compute the I_0^1 integral, if not already computed.
        """
        if not hasattr(self,'I_01'):
            # NB: we pass kh_vectors as a placeholder here; it's not used.
            self.I_01 = simps(self.I_p_q1q2_integrand(0,1,0,self.k_vectors),self.logM_grid)
        return self.I_01

    def _I_p_q1q2_integrand(self,p,q1,q2,k_vectors):
        """Compute the integrand of the I_p^{q1,q2} function defined in the class description.
        This is done over the log10(M/M_sun) grid defined in the __init__ function.

        Note that this is the same as the integrand for the {}_i J_p^{q1,q2} function (for integrals over a finite mass range).

        It also assumes an integration variable log10(M/Msun)

        Parameters:
        - p: Number of halo profiles to include.
        - q1: Order of the first bias term.
        - q2: Order of the second bias term.
        - k_vectors: Vector of N k values (in 1/Mpc units) for each of the p halo profiles. This should have shape (N,p)
        """
        print('be consistent with argument inputs and h factors.')

        assert type(p)==type(q1)==type(q2)==int

        fourier_profiles = 1.
        if p==0:
            fourier_profiles = 1.
        for j in range(p):
            print('Can we parallelize this?')
            fourier_profiles*=self.halo_physics.halo_profile(self.m_h_grid,np.atleast_2d(k_vectors)[j].reshape(-1,1))
        print('Also should pass these profiles as inputs to do multiple vectors')

        # Compute d(n(M))/d(log10(M))
        dn_dlogm = self._compute_mass_function()

        # Define normalization to get correct unity (with Mpc/h and Msun/h units)
        norm = np.power(self.h,3.*float(p)-3.)

        return dn_dlogm * fourier_profiles * self._return_bias(q1) * self._return_bias(q2) * norm

    def _compute_mass_function(self):
        """Compute the mass function for specified masses if not already computed."""
        if not hasattr(self,'mass_function_grid'):
            self.mass_function_grid = self.mass_function.mass_function(self.m_h_grid)
        return self.mass_function_grid

    def _compute_linear_bias(self):
        """Compute the linear bias function for specified masses if not already computed."""
        if not hasattr(self,'linear_bias_grid'):
            self.linear_bias_grid = self.mass_function.linear_halo_bias(self.m_h_grid)
        return self.linear_bias_grid

    def _return_bias(self,q):
        """Return the q-th order halo bias function for all masses in the self.logM_grid attribute.

        Parameters:
        - q: Order of bias. Setting q = 0 returns unity.
        """
        if q==0:
            return 1.
        elif q==1:
            return self._compute_linear_bias()
        else:
            raise Exception('%-th order bias not yet implemented!'%q)
