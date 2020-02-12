from . import Cosmology, MassFunction, HaloPhysics
import numpy as np
from scipy.integrate import simps

class MassIntegrals:
    """
    Class to compute and store the various mass integrals of the form

    .. math::

        I_p^{q1,q2}(k_1,...k_p) = \\int n(m)b^{(q1)}(m)b^{(q2)}\\frac{m^p}{\\rho_M^p}u(k_1|m)..u(k_p|m)dm

    which are needed to compute the power spectrum model. Here :math:`b^{(q)}` is the q-th order bias (with :math:`b^{0}=1`),
    :math:`u(k|m)` is the normalized halo profile and :math:`n(m)` is the mass function.

    All integrals are performed via Simpson's rule over a specified mass range, and are simply returned if they are already computed.

    For the :math:`I_1^{1,0} = I_1^1` integral, the integral must be corrected to ensure that we recover the bias consistency relation :math:`I_1^1 \\rightarrow 1` as :math:`k \\rightarrow 0`.

    This requires an infinite mass range, so we instead approximate;

    .. math::

        I_1^1(k)_{\mathrm{corrected}} = I_1^1(k) + ( 1 - I_1^1(0) ) \\frac{u(k|m_min)}{u(k|0)}

    for normalized halo profile :math:`u(k|m)`.

    Args:
        cosmology (Cosmology): Instance of the Cosmology class containing relevant cosmology and functions.
        mass_function (MassFunction): Instance of the MassFunction class, containing the mass function and bias.
        halo_physics (HaloPhysics): Instance of the HaloPhysics class, containing the halo profiles and concentrations.
        kh_vector (np.ndarray): Array (or float) of wavenumbers in :math:`h\mathrm{Mpc}^{-1}` units from which to compute mass integrals.

    Keyword Args:
        min_logM (float): Minimum mass in :math:`\log_{10}(M/M_\mathrm{sun})` units, default: 6.001.
        max_logM (float): Maximum mass in :math:`\log_{10}(M/M_\mathrm{sun})` units, default: 16.999.
        N_mass (int): Number of logarithmically spaced mass grid points, default: 10000.

    """
    def __init__(self,cosmology,mass_function,halo_physics,kh_vector,min_logM=6.001, max_logM=16.999, N_mass=int(1e4)):
        """Initialize the class with relevant model hyperparameters.
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

    def compute_I_00(self):
        """Compute the I_0^0 integral, if not already computed.

        Returns:
            float: Value of :math:`I_0^0`
        """
        if not hasattr(self,'I_00'):
            # NB: we pass kh_vectors as a placeholder here; it's not used.
            self.I_00 = simps(self._I_p_q1q2_integrand(0,0,0),self.logM_grid)
        return self.I_00.copy()

    def compute_I_01(self):
        """Compute the I_0^1 integral, if not already computed.

        Returns:
            float: Value of :math:`I_0^1`
        """
        if not hasattr(self,'I_01'):
            # NB: we pass kh_vectors as a placeholder here; it's not used.
            self.I_01 = simps(self._I_p_q1q2_integrand(0,1,0),self.logM_grid)
        return self.I_01.copy()

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
            self.I_11 = simps(self._I_p_q1q2_integrand(1,1,0),self.logM_grid,axis=1)

            if apply_correction:
                A = 1. - simps(self._I_p_q1q2_integrand(1,1,0,zero_k=True),self.logM_grid)
                # compute window functions
                min_m_h = np.power(10.,self.min_logM)*self.h
                min_window = self.halo_physics.halo_profile(min_m_h,self.k_vectors).ravel()
                zero_window = self.halo_physics.halo_profile(min_m_h,0.).ravel()
                self.I_11 += A*min_window/zero_window
        return self.I_11.copy()

    def compute_I_111(self):
        """Compute the :math:`I_1^{1,1}(k)` integral, if not already computed.

        Returns:
            np.ndarray: Array of :math:`I_1^{1,1}` values for each k.
        """
        if not hasattr(self,'I_111'):
            self.I_111 = simps(self._I_p_q1q2_integrand(1,1,1),self.logM_grid,axis=1)
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
            self.I_12 = simps(self._I_p_q1q2_integrand(1,2,0),self.logM_grid)

            if apply_correction:
                A = - simps(self._I_p_q1q2_integrand(1,2,0,zero_k=True),self.logM_grid)
                # compute window functions
                min_m_h = np.power(10.,self.min_logM)*self.h
                min_window = self.halo_physics.halo_profile(min_m_h,self.k_vectors).ravel()
                zero_window = self.halo_physics.halo_profile(min_m_h,0.).ravel()
                self.I_12 += A*min_window/zero_window
        return self.I_12.copy()

    def compute_I_20(self):
        """Compute the :math:`I_2^0(k,k)` integral, if not already computed. Note that we assume both k-vectors are the same here.

        Returns:
            np.ndarray: Array of :math:`I_2^0` values for each k.
        """
        if not hasattr(self,'I_20'):
            self.I_20 = simps(self._I_p_q1q2_integrand(2,0,0),self.logM_grid,axis=1)
        return self.I_20.copy()

    def compute_I_21(self):
        """Compute the :math:`I_2^1(k,k)` integral, if not already computed. Note that we assume both k-vectors are the same here.

        Returns:
            np.ndarray: Array of :math:`I_2^1` values for each k.
        """
        if not hasattr(self,'I_21'):
            self.I_21 = simps(self._I_p_q1q2_integrand(2,1,0),self.logM_grid,axis=1)
        return self.I_21.copy()

    def _I_p_q1q2_integrand(self,p,q1,q2,zero_k=False):
        """Compute the integrand of the :math:`I_p^{q1,q2}` function defined in the class description.
        This is done over the :math:`\log_{10}(M/M_\mathrm{sun}) grid defined in the __init__ function.

        Note that this is the same as the integrand for the :math:`{}_i J_p^{q1,q2}` function (for integrals over a finite mass range).

        It also assumes an integration variable :math:`\log_{10}(M/M_\mathrm{sun}) and integrates for each k in the k-vector specified in the class definition.
        If zero_k is set, it returns the value of the integrand at :math:`k = 0` instead.

        Args:
            p (int): Number of halo profiles to include.
            q1 (int): Order of the first bias term.
            q2 (int): Order of the second bias term.
        -
        Keyword Args:
            zero_k (bool): If True, return the integral ewvaluated at k = 0. Default: False.
        """
        print('be consistent with argument inputs and h factors.')

        assert type(p)==type(q1)==type(q2)==int

        fourier_profiles = 1.
        if p==0:
            fourier_profiles = 1.
        else:
            if zero_k:
                fourier_profiles = np.power(self.halo_physics.halo_profile(self.m_h_grid,-1,norm_only=True),p)
            else:
                fourier_profiles = np.power(self._compute_fourier_profile(),p)

        # Compute d(n(M))/d(log10(M))
        dn_dlogm = self._compute_mass_function()

        # Define normalization to get correct unity (with Mpc/h and Msun/h units)
        norm = np.power(self.h,3.*float(p)-3.)

        return dn_dlogm * fourier_profiles * self._return_bias(q1) * self._return_bias(q2) * norm

    def _compute_fourier_profile(self):
        """
        Compute the normalized halo profile :math:`u(k|m)` for specified masses and k if not already computed.

        Returns:
            np.ndarray: array of :math:`u(k|m)` values.
        """
        if not hasattr(self,'fourier_profile'):
            self.fourier_profile = self.halo_physics.halo_profile(self.m_h_grid,self.k_vectors)
        return self.fourier_profile.copy()

    def _compute_mass_function(self):
        """Compute the mass function for specified masses if not already computed.

        Returns:
            np.ndarray: :math:`dn/d\log_{10}(M)` array
        """
        if not hasattr(self,'mass_function_grid'):
            self.mass_function_grid = self.mass_function.mass_function(self.m_h_grid)
        return self.mass_function_grid.copy()

    def _compute_linear_bias(self):
        """Compute the linear bias function for specified masses if not already computed.

        Returns:
            np.ndarray: Array of Eulerian linear biases :math:`b_1^L(m)`
        """
        if not hasattr(self,'linear_bias_grid'):
            self.linear_bias_grid = self.mass_function.linear_halo_bias(self.m_h_grid)
        return self.linear_bias_grid.copy()

    def _compute_second_order_bias(self):
        """Compute the second order bias function for specified masses if not already computed.

        Returns:
            np.ndarray: Array of second order Eulerian biases :math:`b_2^L(m)`
        """
        if not hasattr(self,'second_order_bias_grid'):
            self.second_order_bias_grid = self.mass_function.second_order_bias(self.m_h_grid)
        return self.second_order_bias_grid.copy()

    def _return_bias(self,q):
        """Return the q-th order halo bias function for all masses in the self.logM_grid attribute.

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
