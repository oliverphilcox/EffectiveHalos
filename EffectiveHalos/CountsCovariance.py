from . import MassIntegrals,MassFunction,HaloPhysics,Cosmology,HaloModel
import numpy as np
from scipy.interpolate import interp1d,interp2d, InterpolatedUnivariateSpline
import fastpt as FASTPT
from mcfit import P2xi,xi2P
from scipy.special import spherical_jn
from scipy.integrate import simps

class CountsCovariance:
    """
    Class to compute the covariance of cluster counts and the non-linear power spectrum using the halo model of Philcox et al. 2020. We provide routines for both the :math:`N_i`-:math:`N_j` and :math:`N_i`-:math:`P(k)` covariance where :math:`N_i` is the halo count in a mass bin defined by [:math:`m_{\mathrm{low},i}`, :math:`m_{\mathrm{high},i}`]

    In the Effective Halo Model, the covariance between :math:`X` and :math:`Y` is defined as

    .. math::

        \mathrm{cov}(X,Y) = \mathrm{cov}_\mathrm{intrinsic}(X,Y) + \mathrm{cov}_\mathrm{exclusion}(X,Y) + \mathrm{cov}_\mathrm{super-sample}(X,Y).

    The full expressions for the cluster auto-covariance and cross-covariance with the power spectrum are lengthy but can be found in Philcox et al. (2020). These depend on mass function integrals, :math:`I_p^q`, :math:`{}_iJ_p^q` and :math:`{}_iK_p^q[f]` which are computed in the MassIntegrals class for mass bin i, :math:`P_{NL}` is the 1-loop non-linear power spectrum from Effective Field Theory and :math:`W(kR)` is a smoothing window on scale R.

    Args:
        cosmology (Cosmology): Class containing relevant cosmology and functions.
        mass_function (MassFunction): Class containing the mass function and bias.
        halo_physics (HaloPhysics): Class containing the halo profiles and concentrations.
        kh_vector (np.ndarray): Vector of wavenumbers (in :math:`h/\mathrm{Mpc}` units), for which power spectra will be computed.
        mass_bins (np.ndarray): Array of mass bin edges, in :math:`h^{-1}M_\mathrm{sun}` units. Must have length N_bins + 1.
        volume: Volume of the survey in :math:`(h^{-1}\mathrm{Mpc})^3`.

    Keyword Args:
        kh_min: Minimum k vector in the simulation (or survey) region in :math:`h/\mathrm{Mpc}` units. Modes below kh_min are set to zero, default 0.
        pt_type (str): Which flavor of perturbation theory to adopt. Options 'EFT' (linear + 1-loop + counterterm), 'SPT' (linear + 1-loop), 'Linear', default: 'EFT'
        pade_resum (bool): If True, use a Pade resummation of the counterterm :math:`k^2/(1+k^2) P_\mathrm{lin}` rather than :math:`k^2 P_\mathrm{lin}(k)`, default: True
        smooth_density (bool): If True, smooth the density field on scale R, i.e. multiply power by W(kR)^2, default: True
        IR_resum (bool): If True, perform IR resummation on the density field to resum non-perturbative long-wavelength modes, default: True
        npoints (int): Number of mass bins to use in numerical integration, default: 1000
        verb (bool): If true output useful messages througout run-time, default: False.

    """

    def __init__(self,cosmology, mass_function, halo_physics, kh_vector, mass_bins, volume, kh_min=0, pt_type = 'EFT', pade_resum = True, smooth_density = True, IR_resum = True, npoints = 1000, verb=False):
        """
        Initialize the class loading properties from the other classes.
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
        if isinstance(halo_physics, HaloPhysics):
            self.halo_physics = halo_physics
        else:
            raise TypeError('halo_physics input must be an instance of the HaloPhysics class!')

        # Write useful attributes
        self.kh_vector = kh_vector
        self.kh_min = kh_min
        self.mass_bins = mass_bins
        self.N_bins = len(mass_bins)-1
        self.N_k = len(self.kh_vector)
        self.volume = volume
        self.verb = verb
        self.pt_type = pt_type
        self.pade_resum = pade_resum
        self.smooth_density = smooth_density
        self.IR_resum = IR_resum
        self.npoints = npoints

        # Generate a power spectrum class with this k-vector
        self.halo_model = HaloModel(cosmology, mass_function, halo_physics, kh_vector, kh_min,verb=self.verb)

        # Copy in the MassIntegrals class
        self.mass_integrals = self.halo_model.mass_integrals

        if self.cosmology.use_neutrinos:
            if self.verb:
                print("Note: massive neutrinos are not implemented in full, so we assume CDM+baryon power spectra here.")
                print("(This will creates only a (subdominant) percent-level error for typical neutrino masses.)")

        # Run some checks
        assert self.mass_bins[0]>=np.power(10.,self.mass_integrals.min_logM_h), 'Minimum bin must be above MassIntegral limit!'
        assert self.mass_bins[-1]<=np.power(10.,self.mass_integrals.max_logM_h), 'Maximum bin must be below MassIntegral limit!'

        # Compute linear power for the k-vector
        self.linear_power = self.cosmology.compute_linear_power(self.kh_vector,self.kh_min).copy()

    def NP_covariance(self, cs2, R, alpha, sigma2_volume=-1, use_exclusion=True, use_SSC=True):
        """
        Compute the full covariance matrix of cluster counts and the matter power spectrum :math:`N_i, P(k)` as defined in the class description.

        An important parameter is :math:`\sigma^2(V)`, the variance of the (linear) density field across the survey or simulation box region. If this is not specified, it will be computed from the volume of the survey, assuming spherical symmetry. Note that this is rarely a valid assumption in practice.

        Using the parameters 'use_exclusion' and 'use_SSC' the user can choose which parts of the covariance should be returned.

        Args:
            cs2 (float): Squared-speed-of-sound :math:`c_s^2` counterterm in :math:`(h^{-1}\mathrm{Mpc})^2` units. This should be set by fitting the power spectrum model. (Unused if pt_type is not "EFT")
            R (float): Smoothing scale in :math:h^{-1}`\mathrm{Mpc}`. This should be set by fitting the power spectrum model. (Unused if smooth_density = False)
            alpha (float): Dimensionless ratio of the halo exclusion radius to the halo Lagrangian radius. (Unused if use_exclusion = False)

        Keyword Args:
            sigma2_volume (float): The variance of the linear density field across the survey. This will be computed from the survey volume, assuming isotropy, if not provided. (Unused if use_SSC = False)
            use_excluson (bool): Whether to include the halo exclusion terms, default: True
            use_SSC (bool): Whether to include the super-sample covariance (SSC) terms, default: True

        Returns:
            np.ndarray: Two-dimensional array of :math:`\mathrm{cov}(N_i,P(k))` with shape (N_bins, N_k) for N_bins mass bins and N_k power spectrum bins.
        """
        # Compute intrinsic covariance
        covariance = self._compute_intrinsic_NP_covariance(cs2, R)

        # Compute exclusion covariance, if required
        if use_exclusion:
            covariance += self._compute_exclusion_NP_covariance(cs2, R, alpha)

        # Compute SSC covariance, if required
        if use_SSC:
            covariance += self._compute_ssc_NP_covariance(cs2, R, sigma2_volume)

        return covariance

    def NN_covariance(self, cs2, R, alpha, sigma2_volume=-1, use_exclusion=True, use_SSC=True):
        """
        Compute the full covariance matrix of cluster counts :math:`N_i, N_j` as defined in the class description.

        An important parameter is :math:`\sigma^2(V)`, the variance of the (linear) density field across the survey or simulation box region. If this is not specified, it will be computed from the volume of the survey, assuming spherical symmetry. Note that this is rarely a valid assumption in practice.

        Furthermore, note that the :math:`c_s^2` and :math:`R` parameters have only a minor impact on the covariances here, whilst the :math:`\alpha` parameter is important, since it controls halo exclusion.

        Using the parameters 'use_exclusion' and 'use_SSC' the user can choose which parts of the covariance should be returned.

        Args:
            cs2 (float): Squared-speed-of-sound :math:`c_s^2` counterterm in :math:`(h^{-1}\mathrm{Mpc})^2` units. This should be set by fitting the power spectrum model. (Unused if pt_type is not "EFT")
            R (float): Smoothing scale in :math:h^{-1}`\mathrm{Mpc}`. This should be set by fitting the power spectrum model. (Unused if smooth_density = False)
            alpha (float): Dimensionless ratio of the halo exclusion radius to the halo Lagrangian radius. (Unused if use_exclusion = False)

        Keyword Args:
            sigma2_volume (float): The variance of the linear density field across the survey. This will be computed from the survey volume, assuming isotropy, if not provided. (Unused if use_SSC = False)
            use_excluson (bool): Whether to include the halo exclusion terms, default: True
            use_SSC (bool): Whether to include the super-sample covariance (SSC) terms, default: True

        Returns:
            np.ndarray: Two-dimensional array of :math:`\mathrm{cov}(N_i,N_j)` with shape (N_bins, N_bins) for N_bins mass bins.
        """
        # Compute intrinsic covariance
        covariance = self._compute_intrinsic_NN_covariance()

        # Compute exclusion covariance, if required
        if use_exclusion:
            covariance += self._compute_exclusion_NN_covariance(cs2, R, alpha)

        # Compute SSC covariance, if required
        if use_SSC:
            covariance += self._compute_ssc_NN_covariance(sigma2_volume)

        return covariance

    def _compute_intrinsic_NP_covariance(self, cs2, R, return_terms=False):
        """Compute the intrinsic covariance matrix of :math:`N_i,P(k)` as defined in Philcox et al. (2020). This features one-, two- and three-halo terms that are computed separately.

        Args:
            cs2 (float): Squared-speed-of-sound :math:`c_s^2` counterterm in :math:`(h^{-1}\mathrm{Mpc})^2` units. (Unused if pt_type is not "EFT")
            R (float): Smoothing scale in :math:h^{-1}`\mathrm{Mpc}`. This is a free parameter of the model. (Unused if smooth_density = False)

        Keyword Args:
            return_terms (bool): If true, return the one-, two- and three-halo terms separately in addition to the combined covariance.

        Returns:
            np.ndarray: Two-dimensional array of no-SSC :math:`\mathrm{cov}(N_i,P(k))` with shape (N_bins, N_k) for N_bins mass bins and N_k power spectrum bins.
            np.ndarray: One-halo contribution to the covariance (if return_terms = True)
            np.ndarray: Two-halo contribution to the covariance (if return_terms = True)
            np.ndarray: Three-halo contribution to the covariance (if return_terms = True)
        """
        if self.verb: print("Computing intrinsic covariance terms")

        # Compute the non-linear power spectrum with counterterms
        power_model = self.halo_model.non_linear_power(cs2, R, self.pt_type, self.pade_resum, self.smooth_density, self.IR_resum)

        # Compute smoothing window
        W_kR = self.halo_model._compute_smoothing_function(R)

        # Compute second order F_2 convolution term (depends on R so must be recomputed)
        self.PF2P = self._compute_PF2P(R)

        # Compute relevant I_p^q integrals, if not already computed
        if not hasattr(self,'I_11'):
            self.I_11 = self.mass_integrals.compute_I_11(apply_correction = True)

        # Load mass integrals for each bin, if not already computed
        self._load_mass_integrals()

        # Compute iJ_p^q integrals in each mass bin if not already computed
        if self.verb: print("Computing mass integrals")
        if not hasattr(self,'all_iJ_11_array'):
            # Note that we don't apply the I_1^1 correction, since we expect the mass to be finite
            self.all_iJ_11_array = np.asarray([self.all_mass_integrals[n_bin].compute_I_11(apply_correction = False) for n_bin in range(self.N_bins)])
        if not hasattr(self,'all_iJ_20_array'):
            self.all_iJ_20_array = np.asarray([self.all_mass_integrals[n_bin].compute_I_20() for n_bin in range(self.N_bins)])
        if not hasattr(self,'all_iJ_12_array'):
            self.all_iJ_12_array = np.asarray([self.all_mass_integrals[n_bin].compute_I_12(apply_correction=False) for n_bin in range(self.N_bins)])
        if not hasattr(self,'all_iJ_02_array'):
            self.all_iJ_02_array = np.asarray([self.all_mass_integrals[n_bin].compute_I_02() for n_bin in range(self.N_bins)]).reshape(-1,1)

        # Now construct the covariance
        if self.verb: print("Constructing output covariance")
        cov_3h = self.all_iJ_02_array.copy() * self.I_11.copy()**2. *  W_kR**4. * self.halo_model.linear_power**2.
        cov_2h = 2. * self.all_iJ_11_array.copy() * self.I_11.copy() * power_model + 2. * self.I_11.copy() * self.all_iJ_12_array.copy() * self.PF2P.copy()
        cov_1h = self.all_iJ_20_array.copy()

        if return_terms:
            return cov_1h+cov_2h+cov_3h, cov_1h, cov_2h, cov_3h
        else:
            return cov_1h+cov_2h+cov_3h

    def _compute_intrinsic_NN_covariance(self):
        """Compute the intrinsic covariance matrix of :math:`N_i,N_j` as defined in Philcox et al. (2020). This simply contains a one-halo term. Note that there is an additional two-halo covariance term at low mass resulting from the finite volume, that is not usually included.

        Returns:
            np.ndarray: Two-dimensional array of no-SSC :math:`\mathrm{cov}(N_i,N_j)` with shape (N_bins, N_bins) for N_bins mass bins.
        """
        if self.verb: print("Computing intrinsic covariance terms")

        # Load mass integrals for each bin, if not already computed
        self._load_mass_integrals()

        # Compute iJ_p^q integrals in each mass bin if not already computed
        if self.verb: print("Computing mass integrals")
        if not hasattr(self,'all_iJ_00_array'):
            self.all_iJ_00_array = np.asarray([self.all_mass_integrals[n_bin].compute_I_00() for n_bin in range(self.N_bins)])

        return np.diag(self.all_iJ_00_array)*self.volume

    def _compute_exclusion_NP_covariance(self, cs2, R, alpha, return_terms=False):
        """Compute the exclusion covariance matrix of :math:`N_i,P(k)` as defined in Philcox et al. (2020). This features one-, two- and three-halo terms that are computed separately.

        Args:
            cs2 (float): Squared-speed-of-sound :math:`c_s^2` counterterm in :math:`(h^{-1}\mathrm{Mpc})^2` units. (Unused if pt_type is not "EFT")
            R (float): Smoothing scale in :math:h^{-1}`\mathrm{Mpc}`. This is a free parameter of the model. (Unused if smooth_density = False)
            alpha (float): Dimensionless ratio of halo exclusion radius to Lagrangian radius. This must be less than unity.

        Keyword Args:
            return_terms (bool): If true, return the one-, two- and three-halo terms separately in addition to the combined covariance.

        Returns:
            np.ndarray: Two-dimensional array of no-SSC :math:`\mathrm{cov}(N_i,P(k))` with shape (N_bins, N_k) for N_bins mass bins and N_k power spectrum bins.
            np.ndarray: One-halo contribution to the covariance (if return_terms = True)
            np.ndarray: Two-halo contribution to the covariance (if return_terms = True)
            np.ndarray: Three-halo contribution to the covariance (if return_terms = True)
        """
        assert alpha<1., "Halo exclusion radius must be smaller than Lagrangian radius!"

        if self.verb: print("Computing exclusion covariance terms")

        # Compute the non-linear power spectrum with counterterms
        power_model = self.halo_model.non_linear_power(cs2, R, self.pt_type, self.pade_resum, self.smooth_density, self.IR_resum)

        # Compute smoothing window and linear power
        W_kR = self.halo_model._compute_smoothing_function(R)
        linear_power = self.halo_model.linear_power

        # Compute relevant I_p^q integrals, if not already computed
        if not hasattr(self,'I_11'):
            self.I_11 = self.mass_integrals.compute_I_11(apply_correction = True)

        # Load mass integrals for each bin, if not already computed
        self._load_mass_integrals()

        # Compute iJ_p^q integrals in each mass bin if not already computed
        if self.verb: print("Computing mass integrals")
        if not hasattr(self,'all_iJ_01_array'):
            self.all_iJ_01_array = np.asarray([self.all_mass_integrals[n_bin].compute_I_01() for n_bin in range(self.N_bins)]).reshape(-1,1)
        if not hasattr(self,'all_iJ_10_array'):
            # Note that we don't apply the I_1^1 correction, since we expect the mass to be finite
            self.all_iJ_10_array = np.asarray([self.all_mass_integrals[n_bin].compute_I_10(apply_correction = False) for n_bin in range(self.N_bins)])
        if not hasattr(self,'all_iJ_111_array'):
            self.all_iJ_111_array = np.asarray([self.all_mass_integrals[n_bin].compute_I_111() for n_bin in range(self.N_bins)])
        if not hasattr(self,'all_iJ_00_array'):
            self.all_iJ_00_array = np.asarray([self.all_mass_integrals[n_bin].compute_I_00() for n_bin in range(self.N_bins)]).reshape(-1,1)
        if not hasattr(self,'all_iJ_11_array'):
            # Note that we don't apply the I_1^1 correction, since we expect the mass to be finite
            self.all_iJ_11_array = np.asarray([self.all_mass_integrals[n_bin].compute_I_11(apply_correction = False) for n_bin in range(self.N_bins)])

        # Load exclusion mass integrals for each bin, if not already computed
        self._load_exclusion_mass_integrals()

        # Load relevant interpolators for S and P*Theta type integrands
        ## Note that these depend on the cs2, R parameters so must be computed separately
        self.S_NL_interp = self._load_S_interp(cs2, R, non_linear = True)
        self.S_L_interp = self._load_S_interp(cs2, R, non_linear = False)
        self.p_theta_interp = self._load_p_theta_interp(cs2, R)

        # Compute iK_p^q[f] type integrals in each mass bin. Note that these must be recomputed for each choice of alpha.
        iK_Theta_01_array = np.asarray([self.all_exclusion_mass_integrals[n_bin].compute_K_Theta_01(alpha) for n_bin in range(self.N_bins)])
        iK_Theta_10_array = np.asarray([self.all_exclusion_mass_integrals[n_bin].compute_K_Theta_10(alpha) for n_bin in range(self.N_bins)])
        iK_S_01_array = np.asarray([self.all_exclusion_mass_integrals[n_bin].compute_K_S_01(alpha, self.S_L_interp) for n_bin in range(self.N_bins)])
        iK_S_21_array = np.asarray([self.all_exclusion_mass_integrals[n_bin].compute_K_S_21(alpha, self.S_NL_interp) for n_bin in range(self.N_bins)])
        iK_V_11_array = np.asarray([self.all_exclusion_mass_integrals[n_bin].compute_K_V_11(alpha) for n_bin in range(self.N_bins)])
        iK_V_20_array = np.asarray([self.all_exclusion_mass_integrals[n_bin].compute_K_V_20(alpha) for n_bin in range(self.N_bins)])
        iK_PTheta_11_array = np.asarray([self.all_exclusion_mass_integrals[n_bin].compute_K_PTheta_11(alpha, self.p_theta_interp) for n_bin in range(self.N_bins)])

        # Now construct the covariance
        if self.verb: print("Constructing output covariance")

        cov_3h = -2. * self.all_iJ_01_array.copy() * self.I_11.copy()**2. * iK_Theta_01_array * linear_power**2. * W_kR**4.

        cov_2h = -2. * self.all_iJ_10_array.copy() * self.I_11.copy() * iK_Theta_01_array * power_model
        cov_2h += 2. * self.all_iJ_111_array.copy() * self.I_11.copy() * iK_S_01_array * linear_power * W_kR**2.
        cov_2h += -2. * self.I_11.copy() * self.all_iJ_00_array.copy().reshape(-1,1) * iK_V_11_array * power_model
        cov_2h += -2. * self.I_11.copy() * self.all_iJ_01_array.copy() * iK_Theta_10_array * power_model

        cov_1h = - self.all_iJ_00_array.copy().reshape(-1,1) * iK_V_20_array
        cov_1h += - self.all_iJ_01_array.copy() * iK_S_21_array
        cov_1h += -2. * self.all_iJ_10_array.copy() * iK_Theta_10_array
        cov_1h += -2. * self.all_iJ_11_array.copy() * iK_PTheta_11_array

        if return_terms:
            return cov_1h+cov_2h+cov_3h, cov_1h, cov_2h, cov_3h
        else:
            return cov_1h+cov_2h+cov_3h

    def _compute_exclusion_NN_covariance(self, cs2, R, alpha):
        """Compute the exclusion covariance matrix of :math:`N_i,N_j` as defined in Philcox et al. (2020). This features only a one-halo terms (in the large survey volume limit).

        Args:
            cs2 (float): Squared-speed-of-sound :math:`c_s^2` counterterm in :math:`(h^{-1}\mathrm{Mpc})^2` units. (Unused if pt_type is not "EFT")
            R (float): Smoothing scale in :math:h^{-1}`\mathrm{Mpc}`. This is a free parameter of the model. (Unused if smooth_density = False)
            alpha (float): Dimensionless ratio of halo exclusion radius to Lagrangian radius. This must be less than unity.

        Returns:
            np.ndarray: Two-dimensional array of no-SSC :math:`\mathrm{cov}(N_i,N_j)` with shape (N_bins, N_bins) for N_bins mass bins.
        """
        assert alpha<1., "Halo exclusion radius must be smaller than Lagrangian radius!"

        if self.verb: print("Computing exclusion covariance terms")

        # Load relevant interpolator for S-type integrands
        # Note that this depend on the cs2, R parameters so must be computed separately
        self.S_NL_interp = self._load_S_interp(cs2, R, non_linear = True)

        ## Compute the mass integrals

        # Note that these terms are similar to those computed in the MassIntegrals class
        # However, we now have double integrals over mass.
        # We work in this class for simplicity, since there are few integrals of this form.

        ex_matV = np.zeros((self.N_bins,self.N_bins))
        ex_matS = np.zeros((self.N_bins,self.N_bins))

        # Load in MassIntegrals classes
        self._load_mass_integrals()

        for i in range(self.N_bins):
            mi_i = self.all_mass_integrals[i]

            # Load in dn_dm and b(m) for bin i (probably already computed)
            dn_i = mi_i._compute_mass_function().reshape(-1,1)
            b_i = mi_i._compute_linear_bias().reshape(-1,1)
            for j in range(i,self.N_bins):
                mi_j = self.all_mass_integrals[j]

                # Load in dn_dm and b(m) for bin j (probably already computed)
                dn_j = mi_j._compute_mass_function().reshape(1,-1)
                b_j = mi_j._compute_linear_bias().reshape(1,-1)

                # Compute exclusion radius for bins i and j
                R_ex = np.power(3.*(mi_i.m_h_grid)/(4.*np.pi*self.cosmology.rhoM),1./3.).reshape(-1,1)*np.ones((1,self.npoints))
                R_ex += np.power(3.*(mi_j.m_h_grid)/(4.*np.pi*self.cosmology.rhoM),1./3.).reshape(1,-1)*np.ones((self.npoints,1))
                R_ex *= alpha

                S2NL_M = self.S_NL_interp(R_ex)
                Vex_M = 4./3. * np.pi*np.power(R_ex,3.)

                # Now fill up exclusion matrices with numerical integrals
                ex_matS[i,j] = simps(simps(dn_i*dn_j*b_i*b_j*S2NL_M,mi_i.logM_h_grid,axis=0),mi_j.logM_h_grid,axis=0)
                ex_matV[i,j] = simps(simps(dn_i*dn_j*Vex_M,mi_i.logM_h_grid,axis=0),mi_j.logM_h_grid,axis=0)

                # Fill up other components by symmetry
                ex_matS[j,i] = ex_matS[i,j]
                ex_matV[j,i] = ex_matV[i,j]

        # Now compute and return the covariance matrix term
        cov_1h = - (ex_matV + ex_matS) * self.volume

        return cov_1h

    def _compute_ssc_NP_covariance(self, cs2, R, sigma2_volume=-1):
        """Compute the SSC covariance matrix of :math:`N_i,P(k)` as defined in the class description.

        Args:
            cs2 (float): Squared-speed-of-sound :math:`c_s^2` counterterm in :math:`(h^{-1}\mathrm{Mpc})^2` units. (Unused if pt_type is not "EFT")
            R (float): Smoothing scale in :math:h^{-1}`\mathrm{Mpc}`. This is a free parameter of the model. (Unused if smooth_density = False)

        Keyword Args:
            sigma2_volume (float): The variance of the linear density field across the survey. This will be computed from the survey volume, assuming isotropy, if not provided.

        Returns:
            np.ndarray: Two-dimensional array of SSC :math:`\mathrm{cov}(N_i,P(k))` with shape (N_bins, N_k) for N_bins mass bins and N_k power spectrum bins."""

        if self.verb: print("Computing super-sample covariance terms")

        # Compute the N(m) derivative
        if not hasattr(self,'dN_ddelta'):
            self.dN_ddelta = self._compute_dN_ddelta().copy()

        # Compute the P(k) derivative
        self.dP_ddelta = self._compute_dP_ddelta(cs2, R).copy()

        # Compute sigma^2(V)
        if sigma2_volume==-1:
            print("Note: Variance of the linear density field sigma^2(V) not provided. This will be computed assuming the survey volume is isotropic.")
            sigma2_volume = self._compute_sigma2_volume()

        cov_ssc = self.dN_ddelta.reshape(-1,1)*self.dP_ddelta.reshape(1,-1)*sigma2_volume

        return cov_ssc

    def _compute_ssc_NN_covariance(self, sigma2_volume=-1):
        """Compute the SSC covariance matrix of :math:`N_i,N_j` as defined in the class description.

        Keyword Args:
            sigma2_volume (float): The variance of the linear density field across the survey. This will be computed from the survey volume, assuming isotropy, if not provided.

        Returns:
            np.ndarray: Two-dimensional array of SSC :math:`\mathrm{cov}(N_i,N_j)` with shape (N_bins, N_bins) for N_bins mass bins."""

        if self.verb: print("Computing super-sample covariance terms")

        # Compute the N(m) derivative
        if not hasattr(self,'dN_ddelta'):
            self.dN_ddelta = self._compute_dN_ddelta().copy()

        # Compute sigma^2(V)
        if sigma2_volume==-1:
            print("Note: Variance of the linear density field sigma^2(V) not provided. This will be computed assuming the survey volume is isotropic.")
            sigma2_volume = self._compute_sigma2_volume()

        cov_ssc = self.dN_ddelta.reshape(-1,1)*self.dN_ddelta.reshape(1,-1)*sigma2_volume

        return cov_ssc

    def _compute_dN_ddelta(self):
        """Compute the response function :math:`dN(m)/d\delta_b` where :math:`\delta_b` is a long wavelength mode. This is needed for super-sample covariances. The array is simply returned if already computed.

        Returns:
            np.ndarray: Array of :math:`dN(m)/d\delta_b` in each mass bin.
        """
        # Compute derivative if not already computed
        if not hasattr(self,'dN_ddelta'):
            if self.verb: print('Computing halo count response')

            # Compute required mass integral
            if not hasattr(self,'all_iJ_01_array'):
                self.all_iJ_01_array = np.asarray([self.all_mass_integrals[n_bin].compute_I_01() for n_bin in range(self.N_bins)]).reshape(-1,1)

            # Construct derivative
            self.dN_ddelta = self.all_iJ_01_array*self.volume

        return self.dN_ddelta

    def _compute_dP_ddelta(self, cs2, R):
        """Compute the response function :math:`dP(k)/d\delta_b` where :math:`\delta_b` is a long wavelength mode. This is needed for super-sample covariances.

        Args:
            cs2 (float): Squared-speed-of-sound :math:`c_s^2` counterterm in :math:`(h^{-1}\mathrm{Mpc})^2` units. (Unused if pt_type is not "EFT")
            R (float): Smoothing scale in :math:h^{-1}`\mathrm{Mpc}`. This is a free parameter of the model. (Unused if smooth_density = False)

        Returns:
            np.ndarray: Array of :math:`dP(k)/d\delta_b` for each momentum :math:`k`.
        """

        # Compute derivative (must recompute since cs2 and R are free)
        if self.verb: print('Computing power spectrum response')

        # Compute the 1-loop power spectrum model in fine bins for the dilation derivative
        fine_k = np.logspace(min(np.log10(self.kh_vector))-0.1,max(np.log10(self.kh_vector))+0.1,1000)
        fine_halo = HaloModel(self.cosmology,self.mass_function,self.halo_physics,fine_k,self.kh_min)
        fine_pk_nl = fine_halo.non_linear_power(cs2,R,self.pt_type, self.pade_resum, self.smooth_density, self.IR_resum)

        # Compute the dilation derivative
        k_av = 0.5*(fine_k[1:]+fine_k[:-1])
        log_vec = np.zeros_like(fine_k)
        log_vec[fine_pk_nl!=0] = np.log(fine_k[fine_pk_nl!=0]**3.*fine_pk_nl[fine_pk_nl!=0])
        dlnk3P_dlnk = InterpolatedUnivariateSpline(k_av,np.diff(log_vec)/np.diff(np.log(fine_k)),ext=1)(self.kh_vector)

        # Compute relevant I_p^q integrals, if not already computed
        if not hasattr(self,'I_11'):
            self.I_11 = self.mass_integrals.compute_I_11(apply_correction = True)
        if not hasattr(self,'I_12'):
            self.I_12 = self.mass_integrals.compute_I_12(apply_correction = True)
        if not hasattr(self,'I_21'):
            self.I_21 = self.mass_integrals.compute_I_21()

        ## Compute relevant power spectrum components
        # Non-linear power
        P_NL = self.halo_model.non_linear_power(cs2, R,self.pt_type, self.pade_resum, self.smooth_density, self.IR_resum)
        # Linear power with IR resummation if present
        P_L = self.halo_model.non_linear_power(cs2,R,'Linear',self.pade_resum, self.smooth_density, self.IR_resum)
        # One-loop component (i.e. the residual)
        P_one_loop = P_NL - P_L
        # One loop ratio (careful of stability)
        ratio = np.zeros_like(P_NL)
        ratio[P_NL!=0] = P_one_loop[P_NL!=0]/P_NL[P_NL!=0]
        # Full power
        P_full = self.halo_model.halo_model(cs2, R, self.pt_type, self.pade_resum, self.smooth_density, self.IR_resum)

        # Reconstruct output spectrum
        dP_HSV = 2. * self.I_11.copy() * self.I_12.copy() * P_NL + self.I_21.copy() # halo sample variance
        dP_BC = self.I_11.copy()**2. * P_NL * (68./21. + 26./21.*ratio) # beat-coupling
        dP_LD = -1./3. * dlnk3P_dlnk * P_full  # linear dilation

        self.dP_ddelta =  dP_HSV + dP_BC + dP_LD

        return self.dP_ddelta

    def _compute_PF2P(self,R):
        """
        Compute and return the second order convolution term :math:`\int (2\pi)^{-3}d\vec p F_2(\vec p,\vec k-\vec p)P(\vec p)P(\vec k-\vec p)` where :math:`F_2` is the second order perturbation theory density kernel and :math:`P(\vec k)` are (windowed) power spectra. This is computed using FASTPT.

        Arguments:
            R (float): Smoothing scale in :math:\mathrm{Mpc}/h` units.

        Returns:
            np.ndarray: Array of values of the convolution integral.
        """

        # Prepare FASTPT
        if not hasattr(self,'fastpt'):
            min_k = np.max([np.min(self.kh_vector),1e-4]) # setting minimum to avoid zero errors
            max_k = np.min([np.max(self.kh_vector),1e2])
            self.kh_interp = np.logspace(np.log10(min_k)-1,np.log10(max_k)+1,int(1e4))
            # Compute the one-loop spectrum using FAST-PT
            self.fastpt = FASTPT.FASTPT(self.kh_interp,to_do=['dd_bias'],n_pad=len(self.kh_interp)*3);

        # Now compute the smoothing function
        Wk = 3.*(np.sin(self.kh_interp*R)-self.kh_interp*R*np.cos(self.kh_interp*R))/(self.kh_interp*R)**3.

        # Compute the FASPT spectrum and interpolate to output grid
        out=self.fastpt.one_loop_dd_bias((self.cosmology.compute_linear_power(self.kh_interp,self.kh_min)*Wk).copy(),C_window=0.65,P_window=[0.25,0.25])
        PF2P_power = out[2]/2.
        PF2P_int = InterpolatedUnivariateSpline(self.kh_interp,PF2P_power*Wk)

        return PF2P_int(self.kh_vector)

    def _compute_sigma2_volume(self):
        """
        Compute and return the variance of the linear power spectrum on the scale of the survey volume. Here, we assume a periodic survey, such that the volume can be translated into an isotropic radius. Here, :math:`\sigma^2(R)` is computed from CLASS.

        If this has previously been computed, the value is simply returned.

        Returns:
            float: Value of :math:`\sigma^2(V)` for the survey with volume specified in the class description.
        """

        R_survey = np.power(3.*self.volume/(4.*np.pi),1./3.) # equivalent survey volume
        sigma2_volume = np.power(self.cosmology.vector_sigma_R(R_survey),2.)

        return sigma2_volume

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
                min_logM_h = np.log10(self.mass_bins[n_bin])
                max_logM_h = np.log10(self.mass_bins[n_bin+1])

                # Load an instance of the MassIntegral class
                this_mass_integral = MassIntegrals(self.cosmology,self.mass_function,self.halo_physics,self.kh_vector,
                                                min_logM_h=min_logM_h, max_logM_h=max_logM_h, npoints=self.npoints)
                self.all_mass_integrals.append(this_mass_integral)

    def _load_exclusion_mass_integrals(self):
        """Load the instances of the MassIntegrals class for each mass bin. (Note each integral extends from the average of a given bin to infinity).
        These will be used to compute the :math:`{}_iK_p^q` type integrals.

        This is an empty function if these have already been computed.
        """

        if not hasattr(self,'all_exclusion_mass_integrals'):
            self.all_exclusion_mass_integrals = []
            # Iterate over all mass bins
            for n_bin in range(self.N_bins):

                # Compute mass ranges and convert to Msun units
                av_M_h = 0.5*(self.mass_bins[n_bin]+self.mass_bins[n_bin+1])

                # Load an instance of the MassIntegral class
                this_mass_integral = MassIntegrals(self.cosmology,self.mass_function,self.halo_physics,self.kh_vector,
                                                min_logM_h=np.log10(av_M_h),npoints=self.npoints)
                self.all_exclusion_mass_integrals.append(this_mass_integral)

    def _load_S_interp(self, cs2, R, non_linear = True):
        """Compute and return an interpolator for the :math:`S` function defined as :math:`S(R_\mathrm{ex}) = \int d\vec x \\xi (\vec x) \Theta(\vec x, R_\mathrm{ex})`.

        Args:
            cs2 (float): Squared-speed-of-sound :math:`c_s^2` counterterm in :math:`(h^{-1}\mathrm{Mpc})^2` units. (Unused if non_linear = False)
            R (float): Smoothing scale in :math:h^{-1}`\mathrm{Mpc}`. (Unused if non_linear = False)

        Keyword Args:
            non_linear: If True, use the non-linear power spectrum to define S, default: True.

        Returns:
            interp1d: Interpolator for :math:`S` as a function of exclusion radius.
        """

        if non_linear:
            if self.verb: print("Computing interpolation grid for non-linear S function")
        else:
            if self.verb: print("Computing interpolation grid for linear S function")

        # Define a k grid
        kk = np.logspace(-4,1,10000)

        # Define a power spectrum
        if not self.smooth_density:
            raise Exception("Power spectrum integrals are unstable without density field smoothing!")

        hm2 = HaloModel(self.cosmology, self.mass_function, self.halo_physics, kk, kh_min = self.kh_min)
        if non_linear:
            power_grid = hm2.non_linear_power(cs2, R, self.pt_type, self.pade_resum, self.smooth_density, self.IR_resum)
        else:
            power_grid = hm2.non_linear_power(cs2,R,'Linear',0,self.smooth_density,self.IR_resum)

        # Define interpolation grid for exclusion radii
        RR = np.linspace(0,500,3000).reshape(-1,1)

        # Compute integrals
        S_tmp = simps(power_grid*kk**2./(2.*np.pi**2.)*4.*np.pi*spherical_jn(1,kk*RR)/kk*RR**2.,kk,axis=1)

        return interp1d(RR.ravel(),S_tmp)

    def _load_p_theta_interp(self, cs2, R):
        """Compute and return an interpolator for :math:`\left[P\ast \Theta\right](k,R_\mathrm{ex})` where is an exclusion window function.

        Args:
            cs2 (float): Squared-speed-of-sound :math:`c_s^2` counterterm in :math:`(h^{-1}\mathrm{Mpc})^2` units. (Unused if non_linear = False)
            R (float): Smoothing scale in :math:h^{-1}`\mathrm{Mpc}`. (Unused if non_linear = False)

        Returns:
            interp1d: Interpolator for :math:`\left[P\ast \Theta\right]` as a function of exclusion radius. This is evaluated for all k values.
        """
        if self.verb: print("Computing interpolation grid for P * Theta convolution")


        # Define a k grid
        kk = np.logspace(-4,1,10000)

        # Define a power spectrum
        hm2 = HaloModel(self.cosmology, self.mass_function, self.halo_physics, kk, kh_min = self.kh_min)
        pp = hm2.non_linear_power(cs2, R, self.pt_type, self.pade_resum, self.smooth_density, self.IR_resum)

        # Transform to real space for convolution
        r,xi = P2xi(kk,lowring=False)(pp)

        # Define interpolation grid
        RR = np.linspace(0,200,1000)

        # Multiply in real-space and transform
        xi = np.vstack([xi for _ in range(len(RR))])
        xi[r.reshape(1,-1)>RR.reshape(-1,1)]=0.

        # Interpolate into one dimension and return
        kk,pp = xi2P(r,lowring=False)(xi)
        int2d = interp2d(kk,RR,pp)
        int1d = interp1d(RR,int2d(self.kh_vector,RR).T)
        return lambda rr: int1d(rr.ravel())
