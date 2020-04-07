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

    - **NFW**: Navarro, Frenk & White (1997) universal halo profile.

    Args:
        cosmology (Cosmology): Instance of the Cosmology class containing relevant cosmology and functions.
        mass_function (MassFunction): Instance of the MassFunction class, containing the mass function and bias
        concentration_name (str): Concentration parametrization to use (see above), default: 'Duffy'
        profile_name (str): Halo profile parametrization to use (see above), default: 'NFW'
        hyperparams (kwargs): Any additional parameters to pass to the class (see below).

    Keyword Args:
        min_logM_h (float): Minimum halo mass in :math:`\log_{10}(M/h^{-1}M_\mathrm{sun})`, default: 6
        max_logM_h (float): Maximum halo mass in :math:`\log_{10}(M/h^{-1}M_\mathrm{sun})`, default: 17
        npoints (int): Number of sampling points in mass for :math:`\sigma(M)` interpolation and mass integrals, default: 1e5
        halo_overdensity (float): Characteristic halo overdensity in units of background matter density. Can be a fixed value or 'virial', whereupon the virial collapse overdensity relation of Bryan & Norman 1998 to construct this. Default: 200.
        verb (bool): If true output useful messages througout run-time, default: False.

    """

    def __init__(self,cosmology,mass_function,concentration_name='Duffy',profile_name='NFW',min_logM_h=6,max_logM_h=17,npoints=int(1e5),halo_overdensity=200,verb=False):
        """
        Initialize the class with relevant model hyperparameters.
        """
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
        self.min_logM_h = min_logM_h
        self.max_logM_h = max_logM_h
        self.npoints = npoints

        # Load halo overdensity
        if halo_overdensity=='virial':
            self.halo_overdensity = self._virial_overdensity()
        else:
            self.halo_overdensity = halo_overdensity

        # Create interpolators for sigma and d(ln(sigma))/dlog10(M):
        cosmology._interpolate_sigma_and_deriv(self.min_logM_h,self.max_logM_h,self.npoints)

        # Save reduced Hubble value for later use
        self.a = self.cosmology.a
        self.verb = verb

    def halo_profile(self,m_h,kh,norm_only=False):
        """Compute the halo profile function in Fourier space; :math:`\\rho(k|m) = \\frac{m}{\\bar{\\rho}_M}u(k|m)`
        where :math:`\\bar{\\rho}_M`` is the background matter density and :math:`u(k|m)` is the halo profile.

        We assume halos have a virial collapse overdensity here, based on the parametrization of Bryan & Norman 1998.

        For details of the available profile parametrizations, see the class description.

        Args:
            m_h (np.ndarray): Mass in :math:`h^{-1}M_\mathrm{sun}` units.
            kh (np.ndarray): Wavenumber in h/Mpc units.
            norm_only (bool): Boolean, if set, just return the normalization factor :math:`m/\\bar{\\rho}_M`, default: False

        Returns:
            np.ndarray: Halo profile :math:`\\rho(k|m)` or :math:`m/\\bar{\\rho}_M`, if the norm_only parameter is set.
        """
        if norm_only:
            return m_h/self.cosmology.rhoM

        if self.profile_name=='NFW':
            # Compute overdensity
            odelta = self.halo_overdensity

            # The halo virial radius in Mpc/h units
            rv = np.power(m_h*3.0/(4.0*np.pi*self.cosmology.rhoM*odelta),1.0/3.0)

            # Compute halo concentration
            c = self.halo_concentration(m_h);
            # The function u is normalised to 1 for k<<1 so multiplying by M/rho turns units to a density in units normalized by h
            return self._normalized_halo_profile(kh,rv, c)*m_h/self.cosmology.rhoM;

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

        if self.concentration_name=='Duffy':
            m_h_pivot = 2e12;
            return 7.85*np.power(m_h/m_h_pivot,-0.081)*pow(self.a,0.71);
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

    def _normalized_halo_profile(self,k_h,r_virial,c):
        """Compute the normalized halo profile function in Fourier space; :math:`u(k|m)`

        For details of the available profile parametrizations, see the class description.

        Note that the function returns unity for :math:`k \\leq 0`.

        Args:
            k_h (np.ndarray): Wavenumber in h/Mpc units.
            r_virial (np.ndarray): Virial radius in Mpc/h units.
            c (np.ndarray): Halo concentration parameter; :math:`c = r_\mathrm{virial}/r_\mathrm{scale}`.

        Returns:
            np.ndarray: Normalized halo profile :math:`u(k|m)`
        """

        if self.profile_name=='NFW':

            r_scale = r_virial/c # in Mpc/h units

            # Check if we have an array or a float input and compute accordingly
            if type(k_h)==np.ndarray:
                # filter out sections with k_h<=0
                filt = np.where(k_h>0)

                # compute the matrix of r_scale * k_h
                ks0 = np.matmul(k_h.reshape(-1,1),r_scale.reshape(1,-1))
                ks = ks0[filt,:]
            else:
                if k_h<=0.:
                    return 1.
                ks = k_h*r_scale

            sici1 = sici(ks);
            sici2 = sici(ks*(1.+c))
            f1 = np.sin(ks)*(sici2[0]-sici1[0]);
            f2 = np.cos(ks)*(sici2[1]-sici1[1]);
            f3 = np.sin(c*ks)/(ks*(1.+c));
            fc = np.log(1.+c)-c/(1.+c);

            if type(k_h)==np.ndarray:
                output = np.ones_like(ks0)
                output[filt,:]=((f1+f2-f3)/fc)
                return output
            else:
                return (f1+f2-f3)/fc
        else:
            raise NameError('Halo profile %s is not implemented yet'%(self.profile_name))
