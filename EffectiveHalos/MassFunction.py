from . import Cosmology
import numpy as np
from scipy.special import gamma as gamma_fn
from scipy.interpolate import interp1d

class MassFunction:
    """Class to hold a mass function for halos and associated bias.

    Implemented Mass Functions:

    - **Sheth-Tormen**: Sheth & Tormen 1999 analytic mass function. This assumes virialized halos, and uses the critical density from Nakamura & Suto 1997.
    - **Tinker**: Tinker et al. 2010, eq. 8. This assume a spherical overdensity, the value of which can be specified.
    - **Crocce**: Crocce et al. 2009, eq. 22. Calibrated from Friends-of-Friends halos with linking length 0.2
    - **Bhattacharya**: Bhattacharya et al. 2010, eq. 12. Calibrated from Friends-of-Friends halos with linking length 0.2

    Implemented Bias Functions:

    - **Sheth-Tormen** Sheth-Tormen 2001, eq. 8 Associated to the 'Sheth-Tormen' mass function.
    - **Tinker**: Tinker et al. 2010, eq. 15. Associated to the 'Tinker' mass function.
    - **Crocce**: Peak-background split derivation from the 'Crocce' mass function of eq. 22, Crocce et al. 2009.
    - **Bhattacharya**: Bhattacharya et al. 2010, eq. 18. Associated to the 'Bhattacharya' mass function.

    Args:
        cosmology (Cosmology): Class instance containing relevant cosmological information
        mass_function_name (str): Mass function to use (see above), default: 'Crocce'
        mass_param (kwargs): Any additional parameters to pass to the class. These include:
            - tinker_overdensity: (Only for the Tinker mass function): spherical overdensity defining halos, default: 200

    Keyword Args:
        verb (bool): If true output useful messages througout run-time, default: False.

    """

    def __init__(self,cosmology,mass_function_name='Bhattacharya',verb=False,**mass_params):
        """
        Initialize the class by loading the relevant model parameters.
        """

        # Write attributes, if they're of the correct type
        self.mass_function_name = mass_function_name
        if isinstance(cosmology,Cosmology):
            self.cosmology = cosmology
        else:
            raise NameError('cosmology attribute must be an instance of the Cosmology class!')

        # Expansion parameter
        self.a = 1./(1.+self.cosmology.z)

        self.verb = verb

        # Set hyperparameters to default if not specified
        mass_dict = dict(**mass_params)
        if mass_function_name=='Tinker':
            if 'tinker_overdensity' not in mass_dict.keys():
                mass_dict['tinker_overdensity']=200.

        # Define mass-function specific parameters:
        self._load_mass_function_parameters(mass_dict)

    def mass_function(self,m_h):
        """
        Return the mass function, equal to the number density of halos per unit logarithmic mass interval. This assumes the existence of a universal mass function, with

        .. math::

            dn/d\log_{10}M = f(\sigma(M))\\frac{\\bar{\\rho}_M}{M} \\frac{d\ln\sigma(M)}{d\log_{10}(M)}

        where :math:`f` is the universal mass function, :math:`\\bar{\\rho}_M` is the mean matter density at redshift zero and :math:`\sigma^2(M)` is the overdensity variance on spheres with a radius given by the Lagrangian radius for mass :math:`M`.

        For details of the available mass function parametrizations, see the class description.

        Note:
            For efficiency, we return the mass function :math:`\\frac{dn}{d\log_{10}(M)}` rather than the standard form :math:`\\frac{dn}{dM}`.

        Args:
            m_h (np.ndarray): Array of masses in :math:`h^{-1}M_\mathrm{sun}` units.

        Returns:
            np.ndarray: Mass function, :math:`dn/d\log_{10}(M/h^{-1}M_\mathrm{sun})` in :math:`h^3\mathrm{Mpc}^{-3}` units

        """

        logM_h = np.log10(m_h) # log10(M/h^{-1}Msun)
        dlns_dlogM = self.cosmology.dlns_dlogM_int(logM_h)

        # Compute peak height
        sigma = self.cosmology.sigma_logM_int(logM_h)
        nu = self.delta_c/sigma

        # Compute universal mass function f(nu)
        if self.mass_function_name=='Sheth-Tormen':
            f = nu * self.A_ST * (1. + (self.a_ST * nu**2)**(-self.p_ST)) * np.exp(-self.a_ST * nu**2/2.)
        elif self.mass_function_name=='Tinker':
            f = self.alpha*(1.+np.power(self.beta*nu,-2.*self.phi))*pow(nu,2.*self.eta)*np.exp(-self.gamma*nu**2/2.)
        elif self.mass_function_name=='Crocce':
            f = self.pA*(np.power(sigma,-self.pa)+self.pb)*np.exp(-self.pc/sigma**2)
        elif self.mass_function_name=='Bhattacharya':
            f = self.A0 * np.sqrt(2./np.pi)*np.exp(-self.a0*nu**2./2.)*(1.+(self.a0*nu**2.)**-self.p0)*np.power(self.a0*nu**2.,self.q0/2.)

        # Return mass function
        mf = f * self.cosmology.rhoM * dlns_dlogM / m_h
        return mf

    def linear_halo_bias(self,m_h):
        """
        Compute the linear halo bias, from the peak background split.

        Associated bias functions are available for each mass function, and more can be user-defined. See the class description for details of the loaded parametrizations.

        Args:
            m_h (np.ndarray): Array of masses in :math:`h^{-1}M_\mathrm{sun}` units.

        Returns:
            np.ndarray: Linear Eulerian halo bias (dimensionless)
        """

        logM_h = np.log10(m_h)
        sigma= self.cosmology.sigma_logM_int(logM_h);
        nu = self.delta_c/sigma;

        if self.mass_function_name=='Sheth-Tormen':
            return 1.+(self.a_ST*np.power(nu,2)-1.+2.*self.p_ST/(1.+np.power(self.a_ST*np.power(nu,2),self.p_ST)))/self.delta_c;

        elif self.mass_function_name=='Tinker':
            return 1.-self.fit_A*np.power(nu,self.fit_a)/(np.power(nu,self.fit_a)+np.power(self.delta_c,self.fit_a))+self.fit_B*np.power(nu,self.fit_b)+self.fit_C*np.power(nu,self.fit_c);

        elif self.mass_function_name=='Crocce':
            return 1.-self.pa/(self.delta_c + self.pb*self.delta_c*(self.delta_c/nu)**self.pa) + (2.*self.pc*np.power(nu,2.))/np.power(self.delta_c,3.)

        elif self.mass_function_name=='Bhattacharya':
            return 1. + (self.a0*nu**2.+2.*self.p0/(1.+np.power(self.a0*nu**2.,self.p0))-self.q0) / self.delta_c

    def second_order_bias(self,m_h):
        """ Compute the second order Eulerian bias, defined as :math:`\\frac{4}{21}b_1^L + \\frac{1}{2}b_2^L` where :math:`b_1^L` and :math:`b_2^L` are the Lagrangian bias parameters.

        Associated bias functions are available for each mass function, and more can be user-defined. See the class description for details of the loaded parametrizations.

        Args:
        - m_h: Mass in :math:`h^{-1}M_\mathrm{sun}` units.

        Returns:
            np.ndarray: Quadratic Eulerian halo bias (dimensionless)
        """

        logM_h = np.log10(m_h)
        sigma= self.cosmology.sigma_logM_int(logM_h);
        nu = self.delta_c/sigma;

        if self.mass_function_name=='Crocce':
            b1L = -self.pa/(self.delta_c + self.pb*self.delta_c*(self.delta_c/nu)**self.pa) + (2.*self.pc*np.power(nu,2.))/np.power(self.delta_c,3.)
            b2L = (4.*self.pb*self.pc*self.delta_c**2.*(-2.*self.delta_c**2.+self.pc*nu**2.)+nu**self.pa*(self.pa**2.*self.delta_c**4.-4.*(1.+self.pa)*self.pc*self.delta_c**2.*nu**2.+4.*self.pc**2*nu**4.))/(self.delta_c**6.*(self.pb*self.delta_c**self.pa+nu**self.pa))

            b2E = 8./21.*b1L + b2L
            return b2E

        elif self.mass_function_name=='Bhattacharya':
            b1L = (self.a0*nu**2.+2.*self.p0/(1.+np.power(self.a0*nu**2.,self.p0))-self.q0) / self.delta_c
            b2L = (-2*self.a0*nu**2. + self.a0**2.*nu**4. + (4.*self.p0*(self.a0*nu**2. + self.p0 - self.q0))/(1 + (self.a0*nu**2.)**self.p0) - 2.*self.a0*nu**2.*self.q0 + self.q0**2.)/self.delta_c**2.

            b2E = 8./21.*b1L + b2L
            return b2E

        else:
            raise Exception('Second order bias only implemented for the Crocce and Bhattacharya mass functions.')

    def _load_mass_function_parameters(self,mass_dict):
        """Load internal mass function parameters.

        Args:
            mass_function (str): Mass function name (see class description for options).
            mass_dict (dict): Dictionary of additional parameters.

        """

        if self.mass_function_name=='Sheth-Tormen':

            # Load mass function parameters and normalization
            self.p_ST = 0.3
            self.a_ST = 0.707
            self.A_ST  = (1.+2.**(-self.p_ST)*gamma_fn(0.5-self.p_ST)/np.sqrt(np.pi))**(-1.)*np.sqrt(2.*self.a_ST/np.pi)

            # Compute the spherical collapse threshold of Nakamura-Suto, 1997.
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
            self.alpha=interp1d(odeltas,alpha)(odelta)
            beta0=interp1d(odeltas,beta0)(odelta)
            gamma0=interp1d(odeltas,gamma0)(odelta)
            phi0=interp1d(odeltas,phi0)(odelta)
            eta0=interp1d(odeltas,eta0)(odelta)

            self.beta = beta0*self.a**-0.2
            self.phi = phi0*self.a**0.08
            self.eta = eta0*self.a**-0.27
            self.gamma = gamma0*self.a**0.01

            # BIAS FUNCTION PARAMETERS
            y = np.log10(odelta);

            self.fit_A = 1.0 + 0.24*y*np.exp(-np.power(4./y,4.));
            self.fit_a = 0.44*y-0.88;
            self.fit_B = 0.183;
            self.fit_b = 1.5;
            self.fit_C = 0.019+0.107*y+0.19*np.exp(-np.power(4./y,4.));
            self.fit_c = 2.4;

        elif self.mass_function_name=='Crocce':
            self.delta_c = 1.686  # critical density for collapse

            if self.cosmology.name=='Quijote':
                if self.verb: print('Using fitted parameters for the Crocce mass function from Quijote simulations ')
                # Optimal values for the Quijote simulations
                self.pA = 0.729
                self.pa = 2.355
                self.pb = 0.423
                self.pc = 1.318
            else:
                # Optimal values from original simulations
                self.pA = 0.58*self.a**0.13
                self.pa = 1.37*self.a**0.15
                self.pb = 0.3*self.a**0.084
                self.pc = 1.036*self.a**0.024

        elif self.mass_function_name=='Bhattacharya':
            self.delta_c = 1.686 # critical density for collapse

            def compute_A0(p0,q0):
                inv_A0 = (2.**(0.5*(-1. - 2.*self.p0 + self.q0))*(gamma_fn(-self.p0 + self.q0/2.) + 2**self.p0*gamma_fn(self.q0/2.)))/np.sqrt(np.pi)
                return  1./inv_A0

            if self.cosmology.name=='Quijote':
                if self.verb: print('Using fitted parameters for the Bhattacharya mass function from Quijote simulations ')
                # Optimal values for the Quijote simulations
                self.a0 = 0.77403116
                self.p0 = 0.63685683
                self.q0 = 1.66263337
                self.A0 = compute_A0(self.p0,self.q0)

            elif self.cosmology.name=='Abacus':
                if self.verb: print('Using fitted parameters for the Bhattacharya mass function from Abacus simulations ')
                # Optimal values for the Quijote simulations
                self.a0 = 0.86648878
                self.p0 = 1.30206972
                self.q0 = 1.97133804
                self.A0 = 0.35087244
            else:
                # Optimal values from original paper
                self.a0 = 0.788
                self.p0 = 0.807
                self.q0 = 1.795
                self.A0 = compute_A0(self.p0,self.q0)

        else:
            raise NameError('Mass function %s not currently implemented!'%self.mass_function_name)
