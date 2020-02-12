from . import Cosmology,MassFunction
from scipy.interpolate import interp1d
from scipy.special import sici
import numpy as np

class HaloPhysics:
    """
    Class to hold halo model quantities and relevant integrals.

    Implemented Concentration Functions:

    - **Duffy**: Duffy et al. (2008) for virial-density haloes (second section in Table 1)

    Implemented Halo Profiles:

    - **NFW**: Navarro, Frenk & White (1997) universal halo profile. We use the virial collapse overdensity from Bryan & Norman 1998 to construct this.

    Args:
        cosmology (Cosmology): Instance of the Cosmology class containing relevant cosmology and functions.
        mass_function (MassFunction): Instance of the MassFunction class, containing the mass function and bias
        concentration_name (str): Concentration parametrization to use (see above), default: 'Duffy'
        profile_name (str): Halo profile parametrization to use (see above), default: 'NFW'
        hyperparams (kwargs): Any additional parameters to pass to the class (see below).

    Keyword Args:
        logM_min (float): Minimum mass in :math:`\log_{10}(M/M_\mathrm{sun})`, default: 6
        logM_max (float): Maximum mass in :math:`\log_{10}(M/M_\mathrm{sun})`, default: 17
        npoints (int): Number of sampling points for :math:`\sigma(M)` interpolation, default: 1e5
        tinker_overdensity (int): (Only for the Tinker mass function): spherical overdensity defining halos, default: 200
        verb (bool): If true output useful messages througout run-time, default: False.

    """

    def __init__(self,cosmology,mass_function,concentration_name='Duffy',profile_name='NFW',logM_min=6,logM_max=17,npoints=int(1e5),verb=False,tinker_overdensity=200):
        """
        Initialize the class with relevant model hyperparameters.
        """
        print('Should be more consistent with phys / h units')

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
        self.logM_min = logM_min
        self.logM_max = logM_max
        self.npoints = npoints

        # Create interpolators for sigma and d(ln(sigma))/dlog10(M):
        cosmology._interpolate_sigma_and_deriv(self.logM_min,self.logM_max,self.npoints)

        # Save reduced Hubble value for later use
        self.h = self.cosmology.cosmo.h()
        self.a = self.cosmology.a
        self.verb = verb

    def halo_profile(self,m_h,k_phys,norm_only=False):
        """Compute the halo profile function in Fourier space; :math:`\\rho(k|m) = \\frac{m}{\\bar{\\rho}_M}u(k|m)`
        where :math:`\\bar{\\rho}_M`` is the background matter density and :math:`u(k|m)` is the halo profile.

        We assume halos have a virial collapse overdensity here, based on the parametrization of Bryan & Norman 1998.

        For details of the available profile parametrizations, see the class description.

        Args:
            m_h (np.ndarray): Mass in :math:`h^{-1}M_\mathrm{sun}` units.
            k_phys (np.ndarray): Physical wavenumber in 1/Mpc units.
            norm_only (bool): Boolean, if set, just return the normalization factor :math:`m/\\bar{\\rho}_M`, default: False

        Returns:
            np.ndarray: Halo profile :math:`\\rho(k|m)` or :math:`m/\\bar{\\rho}_M`, if the norm_only parameter is set.
        """
        m = m_h/self.h # in Msun units

        if norm_only:
            return m/self.cosmology.rhoM

        if self.profile_name=='NFW':
            # Compute virial overdensity
            print('should this be a virial overdensity?')
            odelta = self._virial_overdensity()

            # The halo virial radius in physical units
            rv = np.power(m*3.0/(4.0*np.pi*self.cosmology.rhoM*odelta),1.0/3.0)

            # Compute halo concentration
            c = self.halo_concentration(m_h);
            # The function u is normalised to 1 for k<<1 so multiplying by M/rho turns units to a density
            return self._normalized_halo_profile(k_phys,rv, c)*m/self.cosmology.rhoM;

        else:
            raise Exception("Halo profile '%s' not currently implemented!"%self.profile_name)

    def halo_concentration(self,m_h):
        """Compute the halo concentration :math:`c = r_\mathrm{virial} / r_\mathrm{scale}`.

        For details of the available concentration parametrizations, see the class description.

        Args:
            m_h (np.ndarray): Mass in :math:`h^{-1}M_\mathrm{sun}` units.

        Returns:
            np.ndarray: Array of concentration parameters.
        """

        m = m_h/self.h

        if self.concentration_name=='Duffy':
            m_pivot = 2e12/self.h;
            return 7.85*np.power(m/m_pivot,-0.081)*pow(self.a,0.71);
        else:
            raise NameError('Concentration profile %s is not implemented yet'%(self.concentration_name))

    def _virial_overdensity(self):
        """Compute the virial collapse overdensity from Bryan-Norman 1998

        Returns:
            float: Virial collapse overdensity
        """
        Om_mz = self.cosmology._Omega_m()
        x = Om_mz-1.;
        Dv0 = 18.*pow(np.pi,2);
        Dv = (Dv0+82.*x-39.*pow(x,2))/Om_mz;

        return Dv;

    def _normalized_halo_profile(self,k_phys,r_virial,c):
        """Compute the normalized halo profile function in Fourier space; :math:`u(k|m)`

        For details of the available profile parametrizations, see the class description.

        Note that the function returns unity for :math:`k \\leq 0`.

        Args:
            k_phys (np.ndarray): Wavenumber in 1/Mpc units.
            r_virial (np.ndarray): Virial radius in Mpc units.
            c (np.ndarray): Halo concentration parameter; :math:`c = r_\mathrm{virial}/r_\mathrm{scale}`.

        Returns:
            np.ndarray: Normalized halo profile :math:`u(k|m)`
        """

        if self.profile_name=='NFW':

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
            f2 = np.cos(ks)*(sici(ks*(1.+c))[1]-sici(ks)[1]);
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
