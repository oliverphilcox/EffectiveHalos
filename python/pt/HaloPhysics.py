from . import Cosmology,MassFunction
from scipy.interpolate import interp1d
from scipy.special import sici
import numpy as np

class HaloPhysics:
    """Class to hold halo model quantities and relevant integrals.

    Implemented Concentration Functions:
    - 'Duffy': Duffy et al. (2008) for virial-density haloes (second section in Table 1)

    Implemented Halo Profiles:
    - 'NFW': Navarro, Frenk & White (1997) universal halo profile. We use the virial collapse overdensity from Bryan & Norman 1998 to construct this.

    Methods:
    - __init__: Initialize the class with loaded modules and choices of halo profile and concentration function.
    - halo_profile: Compute the halo profile in Fourier space, rho(k|M), from a given input M and k-vector.
    - halo_concentration: Compute the concentration parameter for a given mass halo.
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

    def halo_profile(self,m_h,k_phys,norm_only=False):
        """Compute the halo profile function in Fourier space; rho(k|m) = m/rhoM*u(k|m)
        where rhoM is the background matter density and u(k|m) is the halo profile.

        We assume halos have a virial collapse overdensity here.

        For details of the available profile parametrizations, see the class description.

        Parameters:
        - m_h: Mass in Msun/h units.
        - k_phys: Wavenumber in 1/Mpc units.
        - norm_only: Boolean, if set, just return the normalization factor m/rho_M, default: False
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

    def _virial_overdensity(self):
        """Compute the virial collapse overdensity from Bryan-Norman 1998"""
        Om_mz = self.cosmology._Omega_m()
        x = Om_mz-1.;
        Dv0 = 18.*pow(np.pi,2);
        Dv = (Dv0+82.*x-39.*pow(x,2))/Om_mz;

        return Dv;

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
