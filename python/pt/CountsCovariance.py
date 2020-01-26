from . import MassIntegrals,MassFunction,HaloPhysics,Cosmology,HaloPower
import numpy as np
from scipy.interpolate import interp1d

class CountsCovariance:
    """Class to compute the covariance of cluster counts and the non-linear power spectrum
    using the halo model of Philcox et al. 2020

    The model covariance is defined as
        $$ cov_{no SSC}(N_i, P(k)) = I_1^1(k) [ 2 iJ_1^1(k) P_{NL}(k) W^2(kR) + ... ] + iJ_2^0(k,k) $$

    where I_p^q and iJ_p^q are mass function integrals defined in the MassIntegrals class for mass bin i,
    P_{NL} is the 1-loop non-linear power spectrum from Effective Field Theory
    and W(kR) is a smoothing window on scale R. The ellipses refer to 1-loop bispectrum and trispectrum terms
    which are usually ignored.

    N_i is defined as the number of halos in a mass bin defined by [m_{low,i}, m_{high,i}]

    There is an additional contribution from super-sample-covariance:
        $$ cov_{SSC}(N_i, P(k)) = V sigma^2(V) iJ_0^1 [ I_1^1(k) I_1^{1,1}(k) W^2(kR)  + I_2^1(k) ] $$
    where sigma^2(V) is the variance of the linear density field on scales with volume V.
    """

    def __init__(self,cosmology,mass_function,halo_physics,mass_integrals,kh_vector,kh_min,mass_bins,volume):
        """Initialize the class loading properties from the other classes.

        Parameters:
        - cosmology: Instance of the Cosmology class containing relevant cosmology and functions.
        - mass_function: Instance of the MassFunction class, containing the mass function and bias.
        - halo_physics: Instance of the HaloPhysics class, containing the halo profiles and concentrations.
        - mass_integrals: Instance of the MassIntegrals class, containing the mass integrals.
        - kh_vector: Vector of wavenumbers (in h/Mpc units), for which power spectrum will be computed.
        - kh_min: Minimum k vector in the simulation (or survey) region in h/Mpc units. Modes with kh<kh_min are set to zero.
        - mass_bins: Array of mass bin edges, in Msun/h units. Must have length N_bins + 1.
        - volume: Volume of the survey in (Mpc/h)^3. The variance of the linear field will be computed for radius giving this volume
        """
        print('do we really need to specify all these input classes? generate inside class?')

        print('also compute N(m) autocovariance?')

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
        if isinstance(mass_integrals, MassIntegrals):
            self.mass_integrals = mass_integrals
        else:
            raise TypeError('mass_integrals input must be an instance of the MassIntegrals class!')

        # Write useful attributes
        self.kh_vector = kh_vector
        self.kh_min = kh_min
        self.mass_bins = mass_bins
        self.N_bins = len(mass_bins)-1
        self.volume = volume

        # Run some checks
        assert self.mass_bins[0]>=np.power(10.,self.mass_integrals.min_logM)*self.cosmology.h, 'Minimum bin must be above MassIntegral limit!'
        assert self.mass_bins[-1]<=np.power(10.,self.mass_integrals.max_logM)*self.cosmology.h, 'Maximum bin must be below MassIntegral limit!'

        # Compute linear power for the k-vector
        print('should rename cosmology.linear_power to cosmology.compute_linear_power')
        self.linear_power = self.cosmology.linear_power(self.kh_vector,self.kh_min).copy()

        # Generate a power spectrum class with this k-vector
        self.halo_power = HaloPower(cosmology, mass_function, halo_physics, mass_integrals, kh_vector, kh_min)

    def counts_covariance(self, cs2, R, use_SSC=True, pt_type = 'EFT', pade_resum = True, smooth_density = True, IR_resum = True):
        """Compute the full SSC+no-SSC covariance matrix of N_i and P(k) as defined in the class description.

        If use_SSC = False, then we only return the non-SSC covariance.

        Parameters:
        - cs2: Squared-speed-of-sound counterterm in (Mpc/h)^2. (Unused if pt_type is not "EFT")
        - R: Smoothing scale in Mpc/h. This is a free parameter of the model. (Unused if smooth_density = False)
        - use_SSC: Boolean, whether to include the super-sample covariance (SSC) terms, default: True
        - pt_type: Which flavor of perturbation theory to adopt. Options 'EFT' (linear + 1-loop + counterterm), 'SPT' (linear + 1-loop), 'Linear', default: 'EFT'
        - pade_resum: If True, use a Pade resummation of the counterterm (k^2/(1+k^2)) P_lin rather than k^2 P_lin, default: True
        - smooth_density: If True, smooth the density field on scale R, i.e. multiply power by W(kR)^2, default: True
        - IR_resum: If True, perform IR resummation on the density field to resum non-perturbative long-wavelength modes, default: True
        """
        # Compute no-SSC covariance
        covariance = self._compute_no_ssc_covariance(cs2, R, pt_type, pade_resum, smooth_density, IR_resum)
        print('It would probably be nicer to define these parameters on class initialization')

        # Compute SSC covariance, if required
        if use_SSC:
            covariance += self._compute_ssc_covariance(cs2, R, pt_type, pade_resum, smooth_density, IR_resum)

        return covariance

    def _compute_no_ssc_covariance(self, cs2, R, pt_type = 'EFT', pade_resum = True, smooth_density = True, IR_resum = True):
        """Compute the no-SSC covariance matrix of N_i and P(k) as defined in the class description.

        Parameters:
        - cs2: Squared-speed-of-sound counterterm in (Mpc/h)^2. (Unused if pt_type is not "EFT")
        - R: Smoothing scale in Mpc/h. This is a free parameter of the model. (Unused if smooth_density = False)
        - pt_type: Which flavor of perturbation theory to adopt. Options 'EFT' (linear + 1-loop + counterterm), 'SPT' (linear + 1-loop), 'Linear', default: 'EFT'
        - pade_resum: If True, use a Pade resummation of the counterterm (k^2/(1+k^2)) P_lin rather than k^2 P_lin, default: True
        - smooth_density: If True, smooth the density field on scale R, i.e. multiply power by W(kR)^2, default: True
        - IR_resum: If True, perform IR resummation on the density field to resum non-perturbative long-wavelength modes, default: True
        """

        # Compute the non-linear power spectrum
        power_model = self.halo_power.non_linear_power(cs2, R, pt_type, pade_resum, smooth_density, IR_resum)

        # Compute relevant I_p^q integrals, if not already computed
        if not hasattr(self,'I_11'):
            self.I_11 = self.mass_integrals.compute_I_11(apply_correction = True).copy()

        # Load mass integrals for each bin, if not already computed
        self._load_mass_integrals()

        # Compute iJ_p^q integrals in each mass bin and store
        if not hasattr(self,'all_iJ_11_array'):
            # Note that we don't apply the I_1^1 correction, since we expect the mass to be finite
            self.all_iJ_11_array = np.asarray([self.all_mass_integrals[n_bin].compute_I_11(apply_correction = False) for n_bin in range(self.N_bins)])
        if not hasattr(self,'all_iJ_20_array'):
            self.all_iJ_20_array = np.asarray([self.all_mass_integrals[n_bin].compute_I_20() for n_bin in range(self.N_bins)])

        # Now compute the covariance
        two_halo_term = 2. * self.I_11.copy() * self.all_iJ_11_array.copy() * power_model
        one_halo_term = self.all_iJ_20_array.copy()

        return two_halo_term + one_halo_term

    def _compute_ssc_covariance(self, cs2, R, pt_type = 'EFT', pade_resum = True, smooth_density = True, IR_resum = True):
        """Compute the SSC covariance matrix of N_i and P(k) as defined in the class description.

        Parameters:
        - cs2: Squared-speed-of-sound counterterm in (Mpc/h)^2. (Unused if pt_type is not "EFT")
        - R: Smoothing scale in Mpc/h. This is a free parameter of the model. (Unused if smooth_density = False)
        - pt_type: Which flavor of perturbation theory to adopt. Options 'EFT' (linear + 1-loop + counterterm), 'SPT' (linear + 1-loop), 'Linear', default: 'EFT'
        - pade_resum: If True, use a Pade resummation of the counterterm (k^2/(1+k^2)) P_lin rather than k^2 P_lin, default: True
        - smooth_density: If True, smooth the density field on scale R, i.e. multiply power by W(kR)^2, default: True
        - IR_resum: If True, perform IR resummation on the density field to resum non-perturbative long-wavelength modes, default: True
        """

        # Compute the non-linear power spectrum
        power_model = self.halo_power.non_linear_power(cs2, R, pt_type = pt_type, pade_resum = pade_resum, smooth_density = smooth_density, IR_resum = IR_resum)

        # Compute relevant I_p^q integrals, if not already computed
        if not hasattr(self,'I_11'):
            print('no need to load power twice?')
            self.I_11 = self.mass_integrals.compute_I_11(apply_correction = True)
        if not hasattr(self,'I_21'):
            self.I_21 = self.mass_integrals.compute_I_21()
        if not hasattr(self,'I_111'):
            self.I_111 = self.mass_integrals.compute_I_111()

        # Load mass integrals for each bin, if not already computed
        self._load_mass_integrals()

        # Compute iJ_p^q integrals in each mass bin and store
        if not hasattr(self,'all_iJ_01_array'):
            print('allow for more flexible sigma^2(V)?')
            self.all_iJ_01_array = np.asarray([self.all_mass_integrals[n_bin].compute_I_01() for n_bin in range(self.N_bins)])

        # Now compute the covariance
        prefactor = (self.volume * self._compute_sigma2_volume() * self.all_iJ_01_array.copy()).reshape(-1,1)
        three_halo_term = prefactor * (self.I_11.copy() * self.I_111.copy() * power_model).reshape(1,-1)
        two_halo_term = prefactor * self.I_21.copy().reshape(1,-1)

        # return three_halo_term + two_halo_term

        print('NEW SSC Linear Reponse Form - note that this violates consistency for I_12')
        print('Tidy this up + add non-linear terms')

        # Create an interpolator for k^3 P(k)
        if not hasattr(self,'log_derivative'):
            all_k = np.logspace(-3,1,10000)
            all_cal_P = all_k**3.*self.cosmology.linear_power(all_k,0.0)/(2.*np.pi**2.)
            dlogP_dlogk = np.diff(np.log(all_cal_P))/np.diff(np.log(all_k))
            mid_k = 0.5*(all_k[1:]+all_k[:-1])
            R = 68./21. - dlogP_dlogk/3.
            R_interp = interp1d(mid_k,R)
            self.log_derivative = R_interp(self.kh_vector)

        # Compute I_12
        if not hasattr(self,'I_12'):
            self.I_12 = 0.#self.mass_integrals.compute_I_12(apply_correction=True)

        dP_ddelta = (self.I_11**2.*self.log_derivative*self.linear_power+2.*self.I_12*self.I_11*self.linear_power+self.I_21).reshape(1,-1)

        return prefactor * dP_ddelta

    def _compute_sigma2_volume(self):
        """Compute and return the variance of the linear power spectrum on the scale of the survey volume.
        Here, we assume a periodic survey, such that the volume can be translated into an isotropic radius.
        We compute sigma2(R) from CLASS.
        """
        if not hasattr(self,'sigma2_volume'):
            R_survey = np.power(3.*self.volume/(4.*np.pi),1./3.) # equivalent survey volume
            self.sigma2_volume = np.power(self.cosmology.vector_sigma_R(R_survey/self.cosmology.h),2.)

        print('OVERWRITING sigma^2(V)')
        #return 4.717e-4
        return 5.12e-4
        #return self.sigma2_volume.copy()

    def _load_mass_integrals(self):
        """Load the instances of the MassIntegrals class for each mass bin.
        These will be used to compute the iJ_p^q type integrals.

        This is an empty function if these have al ready been computed.
        """

        if not hasattr(self,'all_mass_integrals'):
            self.all_mass_integrals = []
            # Iterate over all mass bins
            for n_bin in range(self.N_bins):

                # Compute mass ranges and convert to Msun units
                min_logM = np.log10(self.mass_bins[n_bin]/self.cosmology.h)
                max_logM = np.log10(self.mass_bins[n_bin+1]/self.cosmology.h)

                print('should keep N_mass free parameter here?')
                # Load an instance of the MassIntegral class
                this_mass_integral = MassIntegrals(self.cosmology,self.mass_function,self.halo_physics,self.kh_vector,
                                                min_logM=min_logM, max_logM=max_logM, N_mass=int(1e3))
                self.all_mass_integrals.append(this_mass_integral)
