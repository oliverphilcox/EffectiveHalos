from . import Cosmology, MassFunction, HaloPhysics
import numpy as np
from scipy.special import spherical_jn
from scipy.integrate import simps

class MassIntegrals:
    """
    Class to compute and store the various mass integrals of the form

    .. math::

        I_p^{q_1,q_2}(k_1,...k_p) = \\int n(m)b^{(q_1)}(m)b^{(q_2)}\\frac{m^p}{\\rho_M^p}u(k_1|m)..u(k_p|m)dm

    which are needed to compute the power spectrum model. Here :math:`b^{(q)}` is the q-th order bias (with :math:`b^{0}=1`),
    :math:`u(k|m)` is the normalized halo profile and :math:`n(m)` is the mass function.

    All integrals are performed via Simpson's rule over a specified mass range, and are simply returned if they are already computed.

    For the :math:`I_1^{1,0} = I_1^1` integral, the integral must be corrected to ensure that we recover the bias consistency relation :math:`I_1^1 \\rightarrow 1` as :math:`k \\rightarrow 0`.

    This requires an infinite mass range, so we instead approximate;

    .. math::

        I_1^1(k)_{\mathrm{corrected}} = I_1^1(k) + ( 1 - I_1^1(0) ) \\frac{u(k|m_min)}{u(k|0)}

    for normalized halo profile :math:`u(k|m)`.

    Note that this can also be used to compute :math:`{}_iJ_p^{q_1,q_2}` and :math:`{}_iK_p^{q_1,q_2}[f]` type integrals required for the exclusion counts covariance.

    Args:
        cosmology (Cosmology): Instance of the Cosmology class containing relevant cosmology and functions.
        mass_function (MassFunction): Instance of the MassFunction class, containing the mass function and bias.
        halo_physics (HaloPhysics): Instance of the HaloPhysics class, containing the halo profiles and concentrations.
        kh_vector (np.ndarray): Array (or float) of wavenumbers in :math:`h\mathrm{Mpc}^{-1}` units from which to compute mass integrals.

    Keyword Args:
        min_logM_h (float): Minimum mass in :math:`\log_{10}(M/h^{-1}M_\mathrm{sun})` units, default: 6.001.
        max_logM_h (float): Maximum mass in :math:`\log_{10}(M/h^{-1}M_\mathrm{sun})` units, default: 16.999.
        npoints (int): Number of logarithmically spaced mass grid points, default: 10000.
        verb (bool): If true output useful messages througout run-time, default: False.
    """
    def __init__(self,cosmology,mass_function,halo_physics,kh_vector,min_logM_h=6.001, max_logM_h=16.999, npoints=int(1e4),verb=False,m_low=-1):
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
        if isinstance(halo_physics, HaloPhysics):
            self.halo_physics = halo_physics
        else:
            raise TypeError('halo_physics input must be an instance of the HaloPhysics class!')

        # Run some important checks
        interp_min = self.halo_physics.min_logM_h
        interp_max = self.halo_physics.max_logM_h
        assert min_logM_h >= interp_min, 'Minimum mass must be greater than the interpolation limit (10^%.2f)'%interp_min
        assert max_logM_h <= interp_max, 'Minimum mass must be less than the interpolation limit (10^%.2f)'%interp_max

        # Save other attributes
        self.min_logM_h = min_logM_h
        self.max_logM_h = max_logM_h
        self.npoints = npoints
        self.verb = verb

        # Define a mass vector for computations
        self.logM_h_grid = np.linspace(self.min_logM_h,self.max_logM_h, self.npoints)

        # Load and convert masses and wavenumbers
        self.m_h_grid = 10.**self.logM_h_grid # in Msun/h

        self.kh_vectors = kh_vector # in h/Mpc
        self.N_k = len(self.kh_vectors) # define number of k points

    def compute_I_00(self):
        """Compute the I_0^0 integral, if not already computed.

        Returns:
            float: Value of :math:`I_0^0`
        """
        if not hasattr(self,'I_00'):
            self.I_00 = simps(self._I_p_q1q2_integrand(0,0,0),self.logM_h_grid)
        return self.I_00.copy()

    def compute_I_01(self):
        """Compute the I_0^1 integral, if not already computed.

        Returns:
            float: Value of :math:`I_0^1`
        """
        if not hasattr(self,'I_01'):
            self.I_01 = simps(self._I_p_q1q2_integrand(0,1,0),self.logM_h_grid)
        return self.I_01.copy()

    def compute_I_02(self):
        """Compute the I_0^2 integral, if not already computed.

        Returns:
            float: Value of :math:`I_0^2`
        """
        if not hasattr(self,'I_02'):
            self.I_02 = simps(self._I_p_q1q2_integrand(0,2,0),self.logM_h_grid)
        return self.I_02.copy()

    def compute_I_10(self, apply_correction = False):
        """Compute the I_1^0 integral, if not already computed. Also apply the correction noted in the class header if required.

        When computing :math:`{}_i J_1^1` type integrals (over a finite mass bin), the correction should *not* be applied.

        Keyword Args:
            apply_correction (bool): Whether to apply the correction in the class header to ensure the bias consistency relation is upheld.

        Returns:
            float: Array of :math:`I_1^0` values for each k.
        """
        if not hasattr(self,'I_10'):
            self.I_10 = simps(self._I_p_q1q2_integrand(1,0,0),self.logM_h_grid)

            if apply_correction:
                A = 1. - simps(self._I_p_q1q2_integrand(1,0,0,zero_k=True),self.logM_h_grid)
                # compute window functions
                min_m_h = np.power(10.,self.min_logM_h)
                min_window = self.halo_physics.halo_profile(min_m_h,self.kh_vectors).ravel()
                zero_window = self.halo_physics.halo_profile(min_m_h,0.).ravel()
                self.I_10 += A*min_window/zero_window
        return self.I_10.copy()

    def compute_I_11(self,apply_correction = True):
        """Compute the :math:`I_1^1(k)` integral, if not already computed. Also apply the correction noted in the class header if required.

        When computing :math:`{}_i J_1^1` type integrals (over a finite mass bin), the correction should *not* be applied.

        Keyword Args:
            apply_correction (bool): Whether to apply the correction in the class header to ensure the bias consistency relation is upheld.

        Returns:
            np.ndarray: Array of :math:`I_1^1` values for each k.
        """

        if not hasattr(self,'I_11'):
            # Compute the integral over the utilized mass range
            self.I_11 = simps(self._I_p_q1q2_integrand(1,1,0),self.logM_h_grid,axis=1)

            if apply_correction:
                A = 1. - simps(self._I_p_q1q2_integrand(1,1,0,zero_k=True),self.logM_h_grid)
                # compute window functions
                min_m_h = np.power(10.,self.min_logM_h)
                min_window = self.halo_physics.halo_profile(min_m_h,self.kh_vectors).ravel()
                zero_window = self.halo_physics.halo_profile(min_m_h,0.).ravel()
                self.I_11 += A*min_window/zero_window
        return self.I_11.copy()

    def compute_I_111(self):
        """Compute the :math:`I_1^{1,1}(k)` integral, if not already computed.

        Returns:
            np.ndarray: Array of :math:`I_1^{1,1}` values for each k.
        """
        if not hasattr(self,'I_111'):
            self.I_111 = simps(self._I_p_q1q2_integrand(1,1,1),self.logM_h_grid,axis=1)
        return self.I_111.copy()

    def compute_I_12(self,apply_correction = True):
        """Compute the :math:`I_1^2(k)` integral, if not already computed. Also apply the correction noted in the class header if required.

        When computing :math:`{}_i J_1^2` type integrals (over a finite mass bin), the correction should *not* be applied.

        Keyword Args:
            apply_correction (bool): Whether to apply the correction in the class header to ensure the bias consistency relation is upheld.

        Returns:
            np.ndarray: Array of :math:`I_1^2` values for each k.
        """
        if not hasattr(self,'I_12'):
            self.I_12 = simps(self._I_p_q1q2_integrand(1,2,0),self.logM_h_grid)

            if apply_correction:
                A = - simps(self._I_p_q1q2_integrand(1,2,0,zero_k=True),self.logM_h_grid)
                # compute window functions
                min_m_h = np.power(10.,self.min_logM_h)
                min_window = self.halo_physics.halo_profile(min_m_h,self.kh_vectors).ravel()
                zero_window = self.halo_physics.halo_profile(min_m_h,0.).ravel()
                self.I_12 += A*min_window/zero_window
        return self.I_12.copy()

    def compute_I_20(self):
        """Compute the :math:`I_2^0(k,k)` integral, if not already computed. Note that we assume both k-vectors are the same here.

        Returns:
            np.ndarray: Array of :math:`I_2^0` values for each k.
        """
        if not hasattr(self,'I_20'):
            self.I_20 = simps(self._I_p_q1q2_integrand(2,0,0),self.logM_h_grid,axis=1)
        return self.I_20.copy()

    def compute_I_21(self):
        """Compute the :math:`I_2^1(k,k)` integral, if not already computed. Note that we assume both k-vectors are the same here.

        Returns:
            np.ndarray: Array of :math:`I_2^1` values for each k.
        """
        if not hasattr(self,'I_21'):
            self.I_21 = simps(self._I_p_q1q2_integrand(2,1,0),self.logM_h_grid,axis=1)
        return self.I_21.copy()

    def _I_p_q1q2_integrand(self,p,q1,q2,zero_k=False):
        """Compute the integrand of the :math:`I_p^{q1,q2}` function defined in the class description.
        This is done over the :math:`\log_{10}(M/h^{-1}M_\mathrm{sun}) grid defined in the __init__ function.

        Note that this is the same as the integrand for the :math:`{}_i J_p^{q1,q2}` function (for integrals over a finite mass range).

        It also assumes an integration variable :math:`\log_{10}(M/h^{-1}M_\mathrm{sun}) and integrates for each k in the k-vector specified in the class definition.
        If zero_k is set, it returns the value of the integrand at :math:`k = 0` instead.

        Args:
            p (int): Number of halo profiles to include.
            q1 (int): Order of the first bias term.
            q2 (int): Order of the second bias term.

        Keyword Args:
            zero_k (bool): If True, return the integral evaluated at k = 0. Default: False.
        """

        assert type(p)==type(q1)==type(q2)==int

        if p==0:
            fourier_profiles = 1.
        else:
            if zero_k:
                fourier_profiles = np.power(self.halo_physics.halo_profile(self.m_h_grid,-1,norm_only=True),p)
            else:
                fourier_profiles = np.power(self._compute_fourier_profile(),p)

        # Compute d(n(M))/d(log10(M/h^{-1}Msun))
        dn_dlogm = self._compute_mass_function()

        return dn_dlogm * fourier_profiles * self._return_bias(q1) * self._return_bias(q2)

    def _compute_fourier_profile(self):
        """
        Compute the halo profile :math:`m / \rho u(k|m)` for specified masses and k if not already computed.

        Returns:
            np.ndarray: array of :math:`m / \rho u(k|m)` values.
        """
        if not hasattr(self,'fourier_profile'):
            self.fourier_profile = self.halo_physics.halo_profile(self.m_h_grid,self.kh_vectors)
        return self.fourier_profile.copy()

    def _compute_mass_function(self):
        """Compute the mass function for specified masses if not already computed.

        Returns:
            np.ndarray: :math:`dn/d\log_{10}(M/h^{-1}M_\mathrm{sun})` array
        """
        if not hasattr(self,'mass_function_grid'):
            self.mass_function_grid = self.mass_function.mass_function(self.m_h_grid)
        return self.mass_function_grid.copy()

    def _compute_linear_bias(self):
        """Compute the linear bias function for specified masses if not already computed.

        Returns:
            np.ndarray: Array of Eulerian linear biases :math:`b_1^E(m)`
        """
        if not hasattr(self,'linear_bias_grid'):
            self.linear_bias_grid = self.mass_function.linear_halo_bias(self.m_h_grid)
        return self.linear_bias_grid.copy()

    def _compute_second_order_bias(self):
        """Compute the second order bias function for specified masses if not already computed.

        Returns:
            np.ndarray: Array of second order Eulerian biases :math:`b_2^E(m)`
        """
        if not hasattr(self,'second_order_bias_grid'):
            self.second_order_bias_grid = self.mass_function.second_order_bias(self.m_h_grid)
        return self.second_order_bias_grid.copy()

    def _return_bias(self,q):
        """Return the q-th order halo bias function for all masses in the self.logM_h_grid attribute.

        Args:
            q (int): Order of bias. Setting q = 0 returns unity. Currently only :math:`q\leq 2` is implemented.

        Returns:
            np.ndarray: Array of q-th order Eulerian biases.
        """
        if q==0:
            return 1.
        elif q==1:
            return self._compute_linear_bias()
        elif q==2:
            return self._compute_second_order_bias()
        else:
            raise Exception('%-th order bias not yet implemented!'%q)

    def _K_p_q1q2_f_integrand(self,p,q1,q2,f_exclusion,alpha):
        """Compute the integrand of the :math:`{}_iK_p^{q1,q2}[f]` functions defined in Philcox et al. (2020).
        This is done over the :math:`\log_{10}(M/h^{-1}M_\mathrm{sun}) grid defined in the __init__ function.

        Note that the upper mass limit should be infinity (or some large value) for these types of integrals.

        It also assumes an integration variable :math:`\log_{10}(M/h^{-1}M_\mathrm{sun}) and integrates for each k in the k-vector specified in the class definition.
        If zero_k is set, it returns the value of the integrand at :math:`k = 0` instead.

        Args:
            p (int): Number of halo profiles to include.
            q1 (int): Order of the first bias term.
            q2 (int): Order of the second bias term.
            f_exclusion (function): Arbitrary function of the exclusion radius :math:`R_\mathrm{ex}` to be included in the integrand.
            alpha (float): Dimensionless ratio of exclusion to Lagrangian halo radius.
        """

        assert type(p)==type(q1)==type(q2)==int

        if p==0:
            fourier_profiles = 1.
        else:
            fourier_profiles = np.power(self._compute_fourier_profile(),p)

        # Compute exclusion radius
        if not hasattr(self,'R_ex'):
            R_ex = np.power(3.*self.m_h_grid/(4.*np.pi*self.cosmology.rhoM),1./3.)
            R_ex += np.power(3.*min(self.m_h_grid)/(4.*np.pi*self.cosmology.rhoM),1./3.)
            self.R_ex = R_ex.reshape(1,-1) * alpha

        # Compute d(n(M))/d(log10(M/h^{-1}Msun))
        dn_dlogm = self._compute_mass_function()

        return dn_dlogm * fourier_profiles * self._return_bias(q1) * self._return_bias(q2) * f_exclusion(self.R_ex)

    def compute_K_Theta_01(self,alpha):
        """Compute the :math:`K_0^1[\Theta](k)` integral, if not already computed. :math:`Theta` is the Fourier transform of the exclusion window function.

        Arguments:
            alpha (float): Dimensionless ratio of exclusion to Lagrangian halo radius

        Returns:
            np.ndarray: Array of :math:`K_0^1[\Theta](k)` values for each k.
        """
        # Define function
        Theta = lambda R: 4.*np.pi*R**2./self.kh_vectors.reshape(-1,1)*spherical_jn(1,self.kh_vectors.reshape(-1,1)*R)

        if not hasattr(self,'K_Theta_01'):
            self.K_Theta_01 = simps(self._K_p_q1q2_f_integrand(0,1,0,Theta,alpha),self.logM_h_grid,axis=1)
        return self.K_Theta_01.copy()

    def compute_K_Theta_10(self,alpha):
        """Compute the :math:`K_1^0[\Theta](k)` integral, if not already computed. :math:`Theta` is the Fourier transform of the exclusion window function.

        Arguments:
            alpha (float): Dimensionless ratio of exclusion to Lagrangian halo radius

        Returns:
            np.ndarray: Array of :math:`K_1^0[\Theta](k)` values for each k.
        """
        # Define function
        Theta = lambda R: 4.*np.pi*R**2./self.kh_vectors.reshape(-1,1)*spherical_jn(1,self.kh_vectors.reshape(-1,1)*R)

        if not hasattr(self,'K_Theta_10'):
            self.K_Theta_10 = simps(self._K_p_q1q2_f_integrand(1,0,0,Theta,alpha),self.logM_h_grid,axis=1)
        return self.K_Theta_10.copy()

    def compute_K_S_01(self,alpha, S_L_interpolator):
        """Compute the :math:`K_0^1[S](k)` integral, if not already computed. :math:`S` is the integral of the 2PCF windowed by the halo exclusion function of radius :math:`R_\mathrm{ex}``.

        Arguments:
            alpha (float): Dimensionless ratio of exclusion to Lagrangian halo radius
            S_L_interpolator (interp1d): Interpolator for the linear :math:`S(R_\mathrm{ex})` function

        Returns:
            np.ndarray: Array of :math:`K_0^1[S](k)` values for each k.
        """

        if not hasattr(self,'K_S_01'):
            self.K_S_01 = simps(self._K_p_q1q2_f_integrand(0,1,0,S_L_interpolator,alpha),self.logM_h_grid,axis=1)
        return self.K_S_01.copy()

    def compute_K_S_21(self,alpha, S_NL_interpolator):
        """Compute the :math:`K_2^1[S](k)` integral, if not already computed. :math:`S` is the integral of the 2PCF windowed by the halo exclusion function of radius :math:`R_\mathrm{ex}``. Note this function uses the non-linear form.

        Arguments:
            alpha (float): Dimensionless ratio of exclusion to Lagrangian halo radius
            S_NL_interpolator (interp1d): Interpolator for the non-linear :math:`S(R_\mathrm{ex})` function

        Returns:
            np.ndarray: Array of :math:`K_2^1[S](k)` values for each k.
        """

        if not hasattr(self,'K_S_21'):
            self.K_S_21 = simps(self._K_p_q1q2_f_integrand(2,1,0,S_NL_interpolator,alpha),self.logM_h_grid,axis=1)
        return self.K_S_21.copy()

    def compute_K_V_11(self,alpha):
        """Compute the :math:`K_1^1[V](k)` integral, if not already computed. :mathrm:`V` is the volume of the exclusion window function.

        Arguments:
            alpha (float): Dimensionless ratio of exclusion to Lagrangian halo radius

        Returns:
            np.ndarray: Array of :math:`K_1^1[V](k)` values for each k.
        """
        # Define function
        V = lambda R: 4.*np.pi*R**3./3.

        if not hasattr(self,'K_V_11'):
            self.K_V_11 = simps(self._K_p_q1q2_f_integrand(1,1,0,V,alpha),self.logM_h_grid,axis=1)
        return self.K_V_11.copy()

    def compute_K_V_20(self,alpha):
        """Compute the :math:`K_2^0[V](k)` integral, if not already computed. :mathrm:`V` is the volume of the exclusion window function.

        Arguments:
            alpha (float): Dimensionless ratio of exclusion to Lagrangian halo radius

        Returns:
            np.ndarray: Array of :math:`K_2^0[V](k)` values for each k.
        """
        # Define function
        V = lambda R: 4.*np.pi*R**3./3.

        if not hasattr(self,'K_V_20'):
            self.K_V_20 = simps(self._K_p_q1q2_f_integrand(2,0,0,V,alpha),self.logM_h_grid,axis=1)
        return self.K_V_20.copy()

    def compute_K_PTheta_11(self,alpha, PTheta_interpolator):
        """Compute the :math:`K_2^1[P \ast \Theta](k)` integral, if not already computed. :math:`\Theta` is the Fourier transform of the exclusion window function which is convolved with the power spectrum.

        Arguments:
            alpha (float): Dimensionless ratio of exclusion to Lagrangian halo radius
            PTheta_interpolator (interp1d): Interpolator for the non-linear :math:`S(R_\mathrm{ex})` function

        Returns:
            np.ndarray: Array of :math:`K_1^1[P\ast \Theta](k)` values for each k.
        """
        if not hasattr(self,'K_PTheta_11'):
            self.K_PTheta_11 = simps(self._K_p_q1q2_f_integrand(1,1,0,PTheta_interpolator,alpha),self.logM_h_grid,axis=1)
        return self.K_PTheta_11.copy()
