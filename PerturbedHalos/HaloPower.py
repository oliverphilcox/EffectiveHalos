from . import Cosmology,MassIntegrals,MassFunction,HaloPhysics
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.integrate import simps
from scipy.special import spherical_jn
import sys
sys.path.append('/home/ophilcox/FAST-PT/')
import FASTPT as FASTPT

class HaloPower:
    """Class to compute the non-linear power spectrum from the halo model of Philcox et al. 2020.

    The model power is defined as

    .. math::

        P_\mathrm{model} = I_1^1(k)^2 P_{NL}(k) W^2(kR) + I_2^0(k,k)

    where :math:`I_p^q` are mass function integrals defined in the MassIntegrals class, :math:`P_{NL}`` is the 1-loop non-linear power spectrum from Effective Field Theory and :math:`W(kR)` is a smoothing window on scale R.

    Args:
        cosmology (Cosmology): Instance of the Cosmology class containing relevant cosmology and functions.
        mass_function (MassFunction): Instance of the MassFunction class, containing the mass function and bias.
        halo_physics (HaloPhysics): Instance of the HaloPhysics class, containing the halo profiles and concentrations.
        mass_integrals (mass_integrals): Instance of the MassIntegrals class, containing the mass integrals.
        kh_vector (float): Vector of wavenumbers (in :math:`h/\mathrm{Mpc}` units), for which power spectrum will be computed.

    Keyword Args:
        kh_min: Minimum k vector in the simulation (or survey) region in :math:`h/\mathrm{Mpc}` units. Modes below kh_min are set to zero, default: 0.

    """

    def __init__(self,cosmology,mass_function,halo_physics,mass_integrals,kh_vector,kh_min=0):
        """Initialize the class loading properties from the other classes.
        """
        print('do we really need to specify all these input classes? - load mass integrals directly')
        print('need to add FAST-PT to path in better way than above')
        print("either take kh-vector from mass_integrals class or don't respecify")

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

        # Compute linear power for the k-vector
        print('should rename cosmology.linear_power to cosmology.compute_linear_power')
        self.linear_power = self.cosmology.linear_power(self.kh_vector).copy()

    def non_linear_power(self,cs2,R,pt_type = 'EFT',pade_resum = True, smooth_density = True, IR_resum = True):
        """
        Compute the non-linear power spectrum to one-loop order, with IR corrections and counterterms. Whilst we recommend including all non-linear effects, these can be optionally removed with the Boolean parameters. Setting (pt_type='Linear', pade_resum=False, smooth_density=False, IR_resum = False) recovers the standard halo model prediction.

        Including all relevant effects, this is defined as

        .. math::

            P_\mathrm{NL}(k, R, c_s^2) = [P_\mathrm{lin}(k) + P_\mathrm{1-loop}(k) + P_\mathrm{counterterm}(k;c_s^2)] W(kR)

        where

        .. math::

            P_\mathrm{counterterm}(k;c_s^2) = - c_s^2 * \\frac{k^2 }{(1 + k^2)} P_\mathrm{lin}(k)

        is the counterterm, and IR resummation is applied to all spectra.

        This computes the relevant integrals if they haven't already been computed. The function returns :math:`P_\mathrm{NL}` given smoothing scale R and effective squared sound-speed :math:`c_s^2`.

        Args:
            cs2 (float): Squared-speed-of-sound counterterm :math:`c_s^2` in :math:`(h^{-1}\mathrm{Mpc})^2` units. (Unused if pt_type is not "EFT")
            R (float): Smoothing scale in :math:`h^{-1}Mpc`. This is a free parameter of the model. (Unused if smooth_density = False)

        Keyword Args:
            pt_type (str): Which flavor of perturbation theory to adopt. Options 'EFT' (linear + 1-loop + counterterm), 'SPT' (linear + 1-loop), 'Linear', default: 'EFT'
            pade_resum (bool): If True, use a Pade resummation of the counterterm :math:`k^2/(1+k^2) P_\mathrm{lin}` rather than :math:`k^2 P_\mathrm{lin}(k)`, default: True
            smooth_density (bool): If True, smooth the density field on scale R, i.e. multiply power by W(kR)^2, default: True
            IR_resum (bool): If True, perform IR resummation on the density field to resum non-perturbative long-wavelength modes, default: True

        Returns:
            float: Non-linear power spectrum :math:`P_\mathrm{NL}` evaluated at the input k-vector.
        """

        if not IR_resum:
            if pt_type=='Linear':
                output = self.linear_power.copy()
            elif pt_type=='SPT':
                output = self.linear_power.copy()+self.compute_one_loop_only_power()
            elif pt_type=='EFT':
                counterterm = -cs2*self.kh_vector**2.*self.linear_power.copy()
                if pade_resum:
                    counterterm/=(1.+self.kh_vector**2.)
                output = self.linear_power.copy()+self.compute_one_loop_only_power()+counterterm
            else:
                raise NameError("pt_type must be 'Linear', 'SPT' or 'EFT'!")
        else:
            self._prepare_IR_resummation()
            if pt_type=='Linear':
                output = self.compute_resummed_linear_power()
            elif pt_type=='SPT':
                output = self.compute_resummed_one_loop_power()
            elif pt_type=='EFT':
                counterterm_tmp = -cs2*self.kh_vector**2.
                if pade_resum:
                    counterterm_tmp/=(1.+self.kh_vector**2.)
                no_wiggle_lin = self.linear_no_wiggle_power
                wiggle_lin = self.linear_power.copy() - no_wiggle_lin
                output = self.compute_resummed_one_loop_power() + counterterm_tmp * (no_wiggle_lin + wiggle_lin * np.exp(-self.BAO_damping*self.kh_vector**2.))
            else:
                raise NameError("pt_type must be 'Linear', 'SPT' or 'EFT'!")

        if smooth_density:
            output *= self._compute_smoothing_function(R)**2.

        return output

    def halo_power(self,cs2,R,pt_type = 'EFT',pade_resum = True, smooth_density = True, IR_resum = True):
        """
        Compute the non-linear halo-model power spectrum to one-loop order, with IR corrections and counterterms. Whilst we recommend including all non-linear effects, these can be optionally removed with the Boolean parameters.

        This is similar to the 'non_linear_power()' function, but includes the halo mass integrals, and is the *complete* model of the matter power spectrum at one-loop-order in our approximations. Note that the function requires two free parameters; the smoothing scale R and the effective squared sound-speed :math:`c_s^2`, which cannot be predicted from theory. (Note that :math:`c_s^2<0` is permissible).

        For further details, see the class description.

        Args:
            cs2 (float): Squared-speed-of-sound counterterm :math:`c_s^2` in :math:`(h^{-1}\mathrm{Mpc})^2` units. (Unused if pt_type is not "EFT")
            R (float): Smoothing scale in :math:`h^{-1}Mpc`. This is a free parameter of the model. (Unused if smooth_density = False)

        Keyword Args:
            pt_type (str): Which flavor of perturbation theory to adopt. Options 'EFT' (linear + 1-loop + counterterm), 'SPT' (linear + 1-loop), 'Linear', default: 'EFT'
            pade_resum (bool): If True, use a Pade resummation of the counterterm :math:`k^2/(1+k^2) P_\mathrm{lin}` rather than :math:`k^2 P_\mathrm{lin}(k)`, default: True
            smooth_density (bool): If True, smooth the density field on scale R, i.e. multiply power by W(kR)^2, default: True
            IR_resum (bool): If True, perform IR resummation on the density field to resum non-perturbative long-wavelength modes, default: True

        Returns:
            float: Non-linear halo model power spectrum :math:`P_\mathrm{halo}` evaluated at the input k-vector.
        """

        # Compute the non-linear power spectrum
        p_non_linear = self.non_linear_power(cs2, R, pt_type, pade_resum, smooth_density, IR_resum)

        # Compute the halo mass function integrals, if not already computed
        if not hasattr(self,'I_11'):
            self.I_11 = self.mass_integrals.compute_I_11(apply_correction = True)
        if not hasattr(self,'I_20'):
            print('separately return 2h + 1h terms?')
            print('this is taking far too long - is anything being recomputed?')
            self.I_20 = self.mass_integrals.compute_I_20()

        # Compute the final mass function
        return p_non_linear*self.I_11.copy()*self.I_11.copy() + self.I_20.copy()

    def compute_one_loop_only_power(self):
        """
        Compute the one-loop SPT power from the linear power spectrum in the Cosmology class. This returns the one-loop power evaluated at the wavenumber vector specfied in the class initialization. When first called, this computes an interpolator function, which is used in this and subsequent calls.

        Returns:
            float: Vector of 1-loop power :math:`P_\mathrm{1-loop}`(k) for the input k-vector.
        """

        if not hasattr(self,'one_loop_only_power'):
            print('should carry over parameters here - or initialize them in the class')
            self.one_loop_only_power = self._one_loop_only_power_interpolater(self.cosmology.linear_power)(self.kh_vector)

        return self.one_loop_only_power.copy()

    def compute_resummed_linear_power(self):
        """
        Compute the IR-resummed linear power spectrum, using the linear power spectrum in the Cosmology class.

        The output power is defined by

        .. math::

            P_\mathrm{lin, IR} = P_\mathrm{lin, nw}(k) + P_\mathrm{lin, w}e^{-k^2\Sigma^2}

        where 'nw' and 'w' refer to the no-wiggle and wiggle parts of the linear power spectrum and :math:`\Sigma^2` is the BAO damping scale (computed in the _prepare_IR_resummation function)

        If already computed, the IR resummed linear power is simply returned.

        Returns:
            float: Vector of IR-resummed linear power :math:`P_\mathrm{lin,IR}(k)` for the input k-vector.
        """

        if not hasattr(self,'resummed_linear_power'):

            # First create IR interpolaters if not present
            self._prepare_IR_resummation()
            print('add ir parameters?')

            # Load no-wiggle and wiggly parts
            no_wiggle = self.linear_no_wiggle_power
            wiggle = self.linear_power - no_wiggle

            # Compute and return IR resummed power
            self.resummed_linear_power = no_wiggle+np.exp(-self.BAO_damping*self.kh_vector**2.)*wiggle

        return self.resummed_linear_power.copy()

    def compute_resummed_one_loop_power(self):
        """
        Compute the IR-resummed linear-plus-one-loop power spectrum, using the linear power spectrum in the Cosmology class.

        The output power is defined by

        .. math::

            P_\mathrm{lin-plus-1-loop, IR} = P_\mathrm{lin, nw}(k) + P_\mathrm{1-loop, nw}(k) + e^{-k^2\Sigma^2} [ P_\mathrm{lin, w}(k) (1 + k^2\Sigma^2) + P_\mathrm{1-loop,w}(k) ]

        where 'nw' and 'w' refer to the no-wiggle and wiggle parts of the linear / 1-loop power spectrum and :math:`Sigma^2` is the BAO damping scale (computed in the _prepare_IR_resummation function)

        Returns:
            float: Vector of IR-resummed linear-plus-one-loop power :math:`P_\mathrm{lin-plus-1-loop,IR}(k)` for the input k-vector.
        """

        if not hasattr(self,'resummed_one_loop_power'):

            # First create IR interpolators if not present
            self._prepare_IR_resummation()
            print('add ir parameters?')

            # Compute 1-loop only power spectrum
            one_loop_all = self.compute_one_loop_only_power()

            # Load no-wiggle and wiggly parts
            no_wiggle_lin = self.linear_no_wiggle_power
            wiggle_lin = self.linear_power - no_wiggle_lin
            no_wiggle_one_loop = self.one_loop_only_no_wiggle_power
            wiggle_one_loop = one_loop_all - no_wiggle_one_loop

            # Compute and return IR resummed power
            self.resummed_one_loop_power = no_wiggle_lin + no_wiggle_one_loop + np.exp(-self.BAO_damping*self.kh_vector**2.)*(wiggle_lin*(1.+self.kh_vector**2.*self.BAO_damping)+wiggle_one_loop)

        return self.resummed_one_loop_power.copy()

    def _compute_smoothing_function(self,R):
            """
            Compute the smoothing function :math:`W(kR)`, for smoothing scale R. This accounts for the smoothing of the density field on scale R and is the Fourier transform of a spherical top-hat of scale R.

            Args:
                R: Smoothing scale in :math:`h^{-1}\mathrm{Mpc}` units.

            Returns:
                float: :math:`W(kR)` evaluated on the input k-vector.
            """
            kR = self.kh_vector*R
            return  3.*(np.sin(kR)-kR*np.cos(kR))/kR**3.

    def _one_loop_only_power_interpolater(self,linear_spectrum, N_interpolate=50,k_cut=3,N_k = 1000):
        """
        Compute the one-loop SPT power interpolator, using the FAST-PT module. This is computed from an input linear power spectrum.

        Note that the FAST-PT output contains large oscillations at high-k. To alleviate this, we perform smoothing interpolation above some k.

        Args:
            linear_spectrum (function): Function taking input wavenumber in h/Mpc units and returning a linear power spectrum.

        Keyword Args:
            N_interpolate (int): Width of smoothing kernel to apply, default: 20.
            k_cut (float): Minimum k (in :math:`h/\mathrm{Mpc}` units) from which to apply smoothing interpolation, default: 3.
            N_k (int): Number of k values used for interpolation.

        Returns:
            scipy.interp1d: An interpolator for the SPT power given an input k (in :math:`h/\mathrm{Mpc}` units).

        """
        print('need nice way of importing FASTPT from user installation')
        print('need nice way of setting interpolation parameters?')
        print('remove interpolation parameters??')

        print('need to test these hyperparameters')

        # Define some k grid for interpolation (with edges well separated from k limits)
        min_k = np.max([np.min(self.kh_vector),1e-4]) # setting minimum to avoid zero errors
        max_k = np.max(self.kh_vector)
        kh_interp = np.logspace(np.log10(min_k)-0.5,np.log10(max_k)+0.5,N_k)

        # Compute the one-loop spectrum using FAST-PT
        fastpt = FASTPT.FASTPT(kh_interp,to_do=['one_loop_dd'],n_pad=len(kh_interp)*3,
                               verbose=0);
        initial_power=fastpt.one_loop_dd(linear_spectrum(kh_interp).copy(),C_window=0.65,P_window=[0.25,0.25])[0]

         # Now convolve k
        filt = kh_interp>k_cut
        convolved_k = np.convolve(kh_interp[filt],np.ones(N_interpolate,)/N_interpolate,mode='valid')
        convolved_power = np.convolve(initial_power[filt],np.ones(N_interpolate,)/N_interpolate,mode='valid')

        # Concatenate to get an output
        combined_power = np.concatenate([initial_power[~filt],convolved_power])
        combined_k = np.concatenate([kh_interp[~filt],convolved_k])

        # Zero any power values with kh < kh_min
        combined_power[combined_k<self.kh_min] = 0.

        # Create and return an interpolator
        return interp1d(combined_k,combined_power)

    def _prepare_IR_resummation(self,N_k=5000,kh_max=1.):
        """
        Compute relevant quantities to allow IR resummation of the non-linear power spectrum to be performed. This computes the no-wiggle power spectrum, from the 4th order polynomial scheme of Hamann et al. 2010.

        A group of spectra for the no-wiggle linear and no-wiggle 1-loop power are output for later use. The BAO damping scale

        .. math::

            \Sigma^2 =  \frac{1}{6\pi^2}\int_0^\Lambda dq\,P_\mathrm{NL}^{nw}(q)\left[1-j_0(q\ell_\mathrm{BAO})+2j_2(q\ell_\mathrm{BAO})\right]

        is also computed.

        This function is empty if spectra and :math:`Sigma^2` have already been computed.

        Keyword Args:
            N_k (int): Number of points over which to compute no-wiggle power spectrum, default: 5000
            kh_max (float): Maximum k (in :math:`h/\mathrm{Mpc}` units) to which to apply the no-wiggle decomposition, default: 1. Beyond k_max, we assume wiggles are negligible, so :math:`P_\mathrm{no-wiggle} = P_\mathrm{full}`]
        """

        if not hasattr(self,'linear_no_wiggle_power') and not hasattr(self,'one_loop_only_no_wiggle_power') and not hasattr(self,'BAO_damping'):

            print('need to test these hyperparameters')

            # First define a k-grid in h/Mpc units
            min_k = np.max([np.min(self.kh_vector),1e-4]) # setting minimum to avoid zero errors
            max_k = np.max(self.kh_vector)
            kh_interp = np.logspace(np.log10(min_k)-0.5,np.log10(max_k)+0.5,N_k)

            # Define turning point of power spectrum (we compute no-wiggle spectrum beyond this point)
            linear_power_interp = self.cosmology.linear_power(kh_interp,kh_min=self.kh_min)
            max_pos = np.where(linear_power_interp==max(linear_power_interp))
            kh_turn = kh_interp[max_pos]
            Pk_turn = linear_power_interp[max_pos]
            Pk_max = self.cosmology.linear_power(np.atleast_1d(kh_max),kh_min=self.kh_min)

            # Define k in required range
            ffilt = np.where(np.logical_and(kh_interp>kh_turn,kh_interp<kh_max))
            kh_filt = kh_interp[ffilt]

            # Compute ln(P(k)) in region
            log_Pk_mid = np.log(linear_power_interp[ffilt])
            logP1 = np.log(Pk_turn)
            logP2 = np.log(Pk_max)

            # Now fit a fourth order polynomial to the data, fixing the values at the edges.
            def _fourth_order_poly(k,coeff):
                a2,a3,a4=coeff
                poly24 = lambda lk: a2*lk**2.+a3*lk**3.+a4*lk**4.
                f1 = logP1 - poly24(np.log(kh_turn))
                f2 = logP2 - poly24(np.log(kh_max))
                a1 = (f1-f2)/(np.log(kh_turn)-np.log(kh_max))
                a0 = f1 - a1*np.log(kh_turn)
                return a0+a1*np.log(k)+poly24(np.log(k))

            def _fourth_order_fit(coeff):
                return ((log_Pk_mid-_fourth_order_poly(kh_interp[ffilt],coeff))**2.).sum()

            poly_fit = minimize(_fourth_order_fit,[0.,0.,0.])

            # Compute the no-wiggle spectrum, inserting the smooth polynomial in the required range
            noWiggleSpec = linear_power_interp
            noWiggleSpec[ffilt] = np.exp(_fourth_order_poly(kh_filt,poly_fit.x))

            # Now compute no-wiggle power via interpolater
            linear_no_wiggle_interp = interp1d(kh_interp,noWiggleSpec)
            self.linear_no_wiggle_power = linear_no_wiggle_interp(self.kh_vector)

            # Compute one-loop interpolator for no-wiggle power
            # This is just the one-loop operator acting on the no-wiggle power spectrum
            print('should add hyperparameters here?')
            self.one_loop_only_no_wiggle_power = self._one_loop_only_power_interpolater(linear_no_wiggle_interp)(self.kh_vector)

            # Compute the BAO smoothing scale Sigma^2
            def _BAO_integrand(q):
                r_BAO = 105. # BAO scale in Mpc/h
                kh_osc = 1./r_BAO
                return self.cosmology.linear_power(q,kh_min=self.kh_min)*(1.-spherical_jn(0,q/kh_osc)+2.*spherical_jn(2,q/kh_osc))/(6.*np.pi**2.)
            kk_grid = np.linspace(1e-4,0.2,10000)

            # Now store the BAO damping scale as Sigma^2
            self.BAO_damping = simps(_BAO_integrand(kk_grid),kk_grid)
            print('Non-linear BAO damping scale = %.2f Mpc/h'%np.sqrt(self.BAO_damping))
