from . import MassIntegrals,MassFunction,HaloPhysics,Cosmology,HaloPower
import numpy as np
from scipy.interpolate import interp1d

class CountsCovariance:
    """
    Class to compute the covariance of cluster counts and the non-linear power spectrum using the halo model of Philcox et al. 2020

    The model covariance is defined as

    .. math::

        \mathrm{cov}_\mathrm{no\,SSC}(N_i, P(k)) = I_1^1(k) [ 2 {}_iJ_1^1(k) P_{NL}(k) W^2(kR) + ... ] + {}_iJ_2^0(k,k)

    where :math:`I_p^q` and :math:`{}_iJ_p^q` are mass function integrals defined in the MassIntegrals class for mass bin i, :math:`P_{NL}` is the 1-loop non-linear power spectrum from Effective Field Theory and :math:`W(kR)` is a smoothing window on scale R. The ellipses refer to 1-loop bispectrum and trispectrum terms which are usually ignored.

    :math:`N_i` is defined as the number of halos in a mass bin defined by [:math:`m_{low,i}`, :math:`m_{high,i}`]

    There is an additional contribution from super-sample-covariance:

    .. math::

        \mathrm{cov}_\mathrm{SSC}(N_i, P(k)) = V \sigma^2(V) {}_iJ_0^1 [ I_1^1(k) I_1^{1,1}(k) W^2(kR)  + I_2^1(k) ]

    where :math:`\sigma^2(V)` is the variance of the linear density field on scales with volume V.

    Args:
        cosmology (Cosmology): Class containing relevant cosmology and functions.
        mass_function (MassFunction): Class containing the mass function and bias.
        halo_physics (HaloPhysics): Class containing the halo profiles and concentrations.
        kh_vector (np.ndarray): Vector of wavenumbers (in :math:`h/\mathrm{Mpc}` units), for which power spectrum will be computed.
        mass_bins (np.ndarray): Array of mass bin edges, in :math:`h^{-1}M_\mathrm{sun}` units. Must have length N_bins + 1.
        volume: Volume of the survey in :math:`(h^{-1}\mathrm{Mpc})^3`. The variance of the linear field will be computed for radius giving this volume

    Keyword Args:
        kh_min: Minimum k vector in the simulation (or survey) region in :math:`h/\mathrm{Mpc}` units. Modes below kh_min are set to zero, default 0.
        verb (bool): If true output useful messages througout run-time, default: False.

    """

    def __init__(self,cosmology,mass_function,halo_physics,kh_vector,mass_bins,volume,kh_min=0,verb=False):
        """
        Initialize the class loading properties from the other classes.
        """
        print('also compute N(m) autocovariance?')
        print('think about how to do sigma^2 + change in class definition.')

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

        # Write useful attributes
        self.kh_vector = kh_vector
        self.kh_min = kh_min
        self.mass_bins = mass_bins
        self.N_bins = len(mass_bins)-1
        self.volume = volume
        self.verb = verb


        # Generate a power spectrum class with this k-vector
        self.halo_power = HaloPower(cosmology, mass_function, halo_physics, kh_vector, kh_min)

        #Copy in the MassIntegrals class
        self.mass_integrals = self.halo_power.mass_integrals

        # Run some checks
        assert self.mass_bins[0]>=np.power(10.,self.mass_integrals.min_logM)*self.cosmology.h, 'Minimum bin must be above MassIntegral limit!'
        assert self.mass_bins[-1]<=np.power(10.,self.mass_integrals.max_logM)*self.cosmology.h, 'Maximum bin must be below MassIntegral limit!'

        # Compute linear power for the k-vector
        self.linear_power = self.cosmology.compute_linear_power(self.kh_vector,self.kh_min).copy()

    def counts_covariance(self, cs2, R, use_SSC=True, pt_type = 'EFT', pade_resum = True, smooth_density = True, IR_resum = True):
        """
        Compute the full SSC+no-SSC covariance matrix of :math:`N_i, P(k)` as defined in the class description. Whilst we recommend including all non-linear effects, these can be optionally removed with the Boolean parameters. Setting (pt_type='Linear', pade_resum=False, smooth_density=False, IR_resum = False) recovers the standard halo model prediction.

        If use_SSC = False, then we only return the non-SSC covariance.

        Args:
            cs2 (float): Squared-speed-of-sound :math:`c_s^2` counterterm in :math:`(h^{-1}\mathrm{Mpc})^2` units. (Unused if pt_type is not "EFT")
            R (float): Smoothing scale in :math:h^{-1}`\mathrm{Mpc}`. This is a free parameter of the model. (Unused if smooth_density = False)

        Keyword Args:
            use_SSC (bool): Whether to include the super-sample covariance (SSC) terms, default: True
            pt_type (str): Which flavor of perturbation theory to adopt. Options 'EFT' (linear + 1-loop + counterterm), 'SPT' (linear + 1-loop), 'Linear', default: 'EFT'
            pade_resum (bool): If True, use a Pade resummation of the counterterm :math:`k^2/(1+k^2) P_\mathrm{lin}` rather than :math:`k^2 P_\mathrm{lin}(k)`, default: True
            smooth_density (bool): If True, smooth the density field on scale R, i.e. multiply power by W(kR)^2, default: True
            IR_resum (bool): If True, perform IR resummation on the density field to resum non-perturbative long-wavelength modes, default: True

        Returns:
            np.ndarray: Two-dimensional array of :math:`\mathrm{cov}(N_i,P(k))` with shape (N_bins, N_k) for N_bins mass bins and N_k power spectrum bins.
        """
        # Compute no-SSC covariance
        covariance = self._compute_no_ssc_covariance(cs2, R, pt_type, pade_resum, smooth_density, IR_resum)
        print('It would probably be nicer to define these parameters on class initialization')

        # Compute SSC covariance, if required
        if use_SSC:
            covariance += self._compute_ssc_covariance(cs2, R, pt_type, pade_resum, smooth_density, IR_resum)

        return covariance

    def _compute_no_ssc_covariance(self, cs2, R, pt_type = 'EFT', pade_resum = True, smooth_density = True, IR_resum = True):
        """Compute the no-SSC covariance matrix of :math:`N_i,P(k)` as defined in the class description.

        Args:
            cs2 (float): Squared-speed-of-sound :math:`c_s^2` counterterm in :math:`(h^{-1}\mathrm{Mpc})^2` units. (Unused if pt_type is not "EFT")
            R (float): Smoothing scale in :math:h^{-1}`\mathrm{Mpc}`. This is a free parameter of the model. (Unused if smooth_density = False)

        Keyword Args:
            pt_type (str): Which flavor of perturbation theory to adopt. Options 'EFT' (linear + 1-loop + counterterm), 'SPT' (linear + 1-loop), 'Linear', default: 'EFT'
            pade_resum (bool): If True, use a Pade resummation of the counterterm :math:`k^2/(1+k^2) P_\mathrm{lin}` rather than :math:`k^2 P_\mathrm{lin}(k)`, default: True
            smooth_density (bool): If True, smooth the density field on scale R, i.e. multiply power by W(kR)^2, default: True
            IR_resum (bool): If True, perform IR resummation on the density field to resum non-perturbative long-wavelength modes, default: True

        Returns:
            np.ndarray: Two-dimensional array of no-SSC :math:`\mathrm{cov}(N_i,P(k))` with shape (N_bins, N_k) for N_bins mass bins and N_k power spectrum bins.
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
        """Compute the SSC covariance matrix of :math:`N_i,P(k)` as defined in the class description.

        Args:
            cs2 (float): Squared-speed-of-sound :math:`c_s^2` counterterm in :math:`(h^{-1}\mathrm{Mpc})^2` units. (Unused if pt_type is not "EFT")
            R (float): Smoothing scale in :math:h^{-1}`\mathrm{Mpc}`. This is a free parameter of the model. (Unused if smooth_density = False)

        Keyword Args:
            pt_type (str): Which flavor of perturbation theory to adopt. Options 'EFT' (linear + 1-loop + counterterm), 'SPT' (linear + 1-loop), 'Linear', default: 'EFT'
            pade_resum (bool): If True, use a Pade resummation of the counterterm :math:`k^2/(1+k^2) P_\mathrm{lin}` rather than :math:`k^2 P_\mathrm{lin}(k)`, default: True
            smooth_density (bool): If True, smooth the density field on scale R, i.e. multiply power by W(kR)^2, default: True
            IR_resum (bool): If True, perform IR resummation on the density field to resum non-perturbative long-wavelength modes, default: True

        Returns:
            np.ndarray: Two-dimensional array of SSC :math:`\mathrm{cov}(N_i,P(k))` with shape (N_bins, N_k) for N_bins mass bins and N_k power spectrum bins."""

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
            all_cal_P = all_k**3.*self.cosmology.compute_linear_power(all_k,0.0)/(2.*np.pi**2.)
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
        """
        Compute and return the variance of the linear power spectrum on the scale of the survey volume. Here, we assume a periodic survey, such that the volume can be translated into an isotropic radius. Here, :math:`\sigma^2(R)` is computed from CLASS.

        If this has previously been computed, the value is simply returned.

        Returns:
            float: Value of :math:`\sigma^2(V)` for the survey with volume specified in the class description.
        """
        if not hasattr(self,'sigma2_volume'):
            R_survey = np.power(3.*self.volume/(4.*np.pi),1./3.) # equivalent survey volume
            self.sigma2_volume = np.power(self.cosmology.vector_sigma_R(R_survey/self.cosmology.h),2.)

        print('OVERWRITING sigma^2(V) - update docstring')
        #return 4.717e-4
        return 5.12e-4

    def _load_mass_integrals(self):
        """Load the instances of the MassIntegrals class for each mass bin.
        These will be used to compute the :math:`{}_iJ_p^q` type integrals.

        This is an empty function if these have already been computed.
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
