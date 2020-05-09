from . import Cosmology,MassIntegrals,MassFunction,HaloPhysics
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.integrate import simps
from scipy.special import spherical_jn
import sys
import fastpt as FASTPT

class HaloModel:
    """Class to compute the non-linear power spectrum from the halo model of Philcox et al. 2020.

    The model power is defined as

    .. math::

        P_\mathrm{model} = I_1^1(k)^2 P_\mathrm{NL}(k) W^2(kR) + I_2^0(k,k)

    where :math:`I_p^q` are mass function integrals defined in the MassIntegrals class, :math:`P_\mathrm{NL}`` is the 1-loop non-linear power spectrum from Effective Field Theory and :math:`W(kR)` is a smoothing window on scale R.

    Args:
        cosmology (Cosmology): Instance of the Cosmology class containing relevant cosmology and functions.
        mass_function (MassFunction): Instance of the MassFunction class, containing the mass function and bias.
        halo_physics (HaloPhysics): Instance of the HaloPhysics class, containing the halo profiles and concentrations.
        kh_vector (np.ndarray): Vector of wavenumbers (in :math:`h/\mathrm{Mpc}` units), for which power spectrum will be computed.

    Keyword Args:
        kh_min: Minimum k vector in the simulation (or survey) region in :math:`h/\mathrm{Mpc}` units. Modes below kh_min are set to zero, default: 0.
        verb (bool): If true, output useful messages througout run-time, default: False.

    """

    def __init__(self,cosmology,mass_function,halo_physics,kh_vector,kh_min=0,verb=False):
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

        # Create instance of the MassIntegrals class
        self.mass_integrals = MassIntegrals(self.cosmology, self.mass_function, self.halo_physics, kh_vector,
                                            min_logM_h = self.halo_physics.min_logM_h+0.01, max_logM_h = self.halo_physics.max_logM_h-0.01,npoints=self.halo_physics.npoints)

        # Write useful attributes
        self.kh_vector = kh_vector
        self.kh_min = kh_min
        self.verb = verb

        # Compute linear (CDM+baryon) power for the k-vector
        self.linear_power = self.cosmology.compute_linear_power(self.kh_vector,self.kh_min).copy()

        # Also compute full (CDM+baryon+neutrino) power if required
        if self.cosmology.use_neutrinos:
            self.linear_power_total = self.cosmology.compute_linear_power(self.kh_vector,self.kh_min,with_neutrinos=True).copy()

        # Set other hyperparameters consistently. (These are non-critical but control minutae of IR resummation and interpolation precision)
        self.IR_N_k = 5000
        self.IR_kh_max = 1.
        self.OneLoop_N_interpolate = 30
        self.OneLoop_k_cut = 3
        self.OneLoop_N_k = 2000

    def non_linear_power(self,cs2,R,pt_type = 'EFT',pade_resum = True, smooth_density = True, IR_resum = True, include_neutrinos = True):
        """
        Compute the non-linear power spectrum to one-loop order, with IR corrections and counterterms. Whilst we recommend including all non-linear effects, these can be optionally removed with the Boolean parameters. Setting (pt_type='Linear', pade_resum=False, smooth_density=False, IR_resum = False) recovers the standard halo model prediction.

        Including all relevant effects, this is defined as

        .. math::

            P_\mathrm{NL}(k, R, c_s^2) = [P_\mathrm{lin}(k) + P_\mathrm{1-loop}(k) + P_\mathrm{counterterm}(k;c_s^2)] W(kR)

        where

        .. math::

            P_\mathrm{counterterm}(k;c_s^2) = - c_s^2 \\frac{k^2 }{(1 + k^2)} P_\mathrm{lin}(k)

        is the counterterm, and IR resummation is applied to all spectra.

        This computes the relevant loop integrals if they haven't already been computed. The function returns :math:`P_\mathrm{NL}` given smoothing scale R and effective squared sound-speed :math:`c_s^2`.

        For massive neutrino cosmologies, we assume that the matter power spectrum is given by a mass-fraction-weighted sum of the non-linear CDM+baryon power spectrum, linear neutrino spectrum and linear neutrino cross CDM+baryon spectrum. This is a good approximation for the halo model spectra (i.e. including non-linear effects only for the CDM+baryon component.) The function can return either the non-linear CDM+baryon power spectrum or the combined CDM+baryon+neutrino power spectrum using the 'include_neutrinos' flag.

        Args:
            cs2 (float): Squared-speed-of-sound counterterm :math:`c_s^2` in :math:`(h^{-1}\mathrm{Mpc})^2` units. (Unused if pt_type is not "EFT")
            R (float): Smoothing scale in :math:`h^{-1}Mpc`. This is a free parameter of the model. (Unused if smooth_density = False)

        Keyword Args:
            pt_type (str): Which flavor of perturbation theory to adopt. Options 'EFT' (linear + 1-loop + counterterm), 'SPT' (linear + 1-loop), 'Linear', default: 'EFT'
            pade_resum (bool): If True, use a Pade resummation of the counterterm :math:`k^2/(1+k^2) P_\mathrm{lin}` rather than :math:`k^2 P_\mathrm{lin}(k)`, default: True
            smooth_density (bool): If True, smooth the density field on scale R, i.e. multiply power by W(kR)^2, default: True
            IR_resum (bool): If True, perform IR resummation on the density field to resum non-perturbative long-wavelength modes, default: True
            include_neutrinos (bool): If True, return the full power spectrum of CDM+baryons+neutrinos (with the approximations given above). If False, return only the CDM+baryon power spectrum. This has no effect in cosmologies without massive neutrinos, default: True.

        Returns:
            np.ndarray: Non-linear power spectrum :math:`P_\mathrm{NL}` evaluated at the input k-vector.
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

        # Now account for neutrino effects if present
        if self.cosmology.use_neutrinos:
            if include_neutrinos:
                f_cb = 1.-self.cosmology.f_nu
                return f_cb**2.*output + (self.linear_power_total.copy()-f_cb**2.*self.linear_power.copy())
            else:
                return output
        else:
            return output

    def halo_model(self,cs2,R,pt_type = 'EFT',pade_resum = True, smooth_density = True, IR_resum = True, include_neutrinos=True, return_terms=False):
        """
        Compute the non-linear halo-model power spectrum to one-loop order, with IR corrections and counterterms. Whilst we recommend including all non-linear effects, these can be optionally removed with the Boolean parameters.

        This is similar to the 'non_linear_power()' function, but includes the halo mass integrals, and is the *complete* model of the matter power spectrum at one-loop-order in our approximations. Note that the function requires two free parameters; the smoothing scale R and the effective squared sound-speed :math:`c_s^2`, which cannot be predicted from theory. (Note that :math:`c_s^2<0` is permissible).

        For massive neutrino cosmologies, we assume that the matter power spectrum is given by a mass-fraction-weighted sum of the halo model CDM+baryon power spectrum, linear neutrino spectrum and linear neutrino cross CDM+baryon spectrum. This is a good approximation in practice (i.e. including halo-model effects only for the CDM+baryon component.) The function can return either the halo-model CDM+baryon power spectrum (suitable for comparison to CDM+baryon power spectra) or the combined CDM+baryon+neutrino power spectrum (suitable for comparison to matter spectra) using the 'include_neutrinos' flag. When 'include_neutrinos' is specified the model has three components; the weighted two-halo CDM+baryon part, the weighted one-halo CDM+baryon part and the weighted linear neutrino and cross spectra. The sum of all three and the first two individually are returned by the 'return_terms' command.

        For further details, see the class description.

        Args:
            cs2 (float): Squared-speed-of-sound counterterm :math:`c_s^2` in :math:`(h^{-1}\mathrm{Mpc})^2` units. (Unused if pt_type is not "EFT")
            R (float): Smoothing scale in :math:`h^{-1}Mpc`. This is a free parameter of the model. (Unused if smooth_density = False)

        Keyword Args:
            pt_type (str): Which flavor of perturbation theory to adopt. Options 'EFT' (linear + 1-loop + counterterm), 'SPT' (linear + 1-loop), 'Linear', default: 'EFT'
            pade_resum (bool): If True, use a Pade resummation of the counterterm :math:`k^2/(1+k^2) P_\mathrm{lin}` rather than :math:`k^2 P_\mathrm{lin}(k)`, default: True
            smooth_density (bool): If True, smooth the density field on scale R, i.e. multiply power by W(kR)^2, default: True
            IR_resum (bool): If True, perform IR resummation on the density field to resum non-perturbative long-wavelength modes, default: True
            include_neutrinos (bool): If True, return the full power spectrum of CDM+baryons+neutrinos (with the approximations given above). If False, return only the CDM+baryon power spectrum. This has no effect in cosmologies without massive neutrinos, default: True.
            return_terms (bool): If True, return the one- and two-halo CDM+baryon halo-model terms in addition to the combined power spectrum model, default: False

        Returns:
            np.ndarray: Non-linear halo model power spectrum :math:`P_\mathrm{halo}` evaluated at the input k-vector.
            np.ndarray: One-halo power spectrum term (if return_terms is true)
            np.ndarray: Two-halo power spectrum term (if return_terms is true)
        """

        # Compute the non-linear power spectrum (neutrinos will be added in later)
        p_non_linear = self.non_linear_power(cs2, R, pt_type, pade_resum, smooth_density, IR_resum, False)

        # Compute the halo mass function integrals, if not already computed
        if not hasattr(self,'I_11'):
            self.I_11 = self.mass_integrals.compute_I_11(apply_correction = True)
        if not hasattr(self,'I_20'):
            self.I_20 = self.mass_integrals.compute_I_20()

        # Compute the final spectrum
        two_halo = p_non_linear*self.I_11.copy()*self.I_11.copy()
        one_halo = self.I_20.copy()
        output_spectrum = two_halo + one_halo

        # Now account for neutrino effects if present
        if self.cosmology.use_neutrinos:
            if include_neutrinos:
                f_cb = 1.-self.cosmology.f_nu
                two_halo *= f_cb**2. # rescale two-halo term
                one_halo *= f_cb**2. # rescale one-halo term
                output_spectrum = f_cb**2.*output_spectrum + (self.linear_power_total.copy()-f_cb**2.*self.linear_power.copy())

        if return_terms:
            return output_spectrum, one_halo, two_halo
        else:
            return output_spectrum

    def compute_one_loop_only_power(self):
        """
        Compute the one-loop SPT power from the linear power spectrum in the Cosmology class. This returns the one-loop power evaluated at the wavenumber vector specfied in the class initialization. When first called, this computes an interpolator function, which is used in this and subsequent calls.

        Returns:
            np.ndarray: Vector of 1-loop power :math:`P_\mathrm{1-loop}(k)` for the input k-vector.
        """

        if not hasattr(self,'one_loop_only_power'):
            self.one_loop_only_power = self._one_loop_only_power_interpolater(lambda kk: self.cosmology.compute_linear_power(kk,self.kh_min))(self.kh_vector)

        return self.one_loop_only_power.copy()

    def compute_resummed_linear_power(self):
        """
        Compute the IR-resummed linear power spectrum, using the linear power spectrum in the Cosmology class.

        The output power is defined by

        .. math::

            P_\mathrm{lin, IR}(k) = P_\mathrm{lin, nw}(k) + P_\mathrm{lin, w}(k)e^{-k^2\Sigma^2}

        where 'nw' and 'w' refer to the no-wiggle and wiggle parts of the linear power spectrum and :math:`\Sigma^2` is the BAO damping scale (computed in the _prepare_IR_resummation function)

        If already computed, the IR resummed linear power is simply returned.

        Returns:
            np.ndarray: Vector of IR-resummed linear power :math:`P_\mathrm{lin,IR}(k)` for the input k-vector.
        """

        if not hasattr(self,'resummed_linear_power'):

            # First create IR interpolaters if not present
            self._prepare_IR_resummation()

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

            P_\mathrm{lin+1, IR}(k) = P_\mathrm{lin, nw}(k) + P_\mathrm{1-loop, nw}(k) + e^{-k^2\Sigma^2} [ P_\mathrm{lin, w}(k) (1 + k^2\Sigma^2) + P_\mathrm{1-loop,w}(k) ]

        where 'nw' and 'w' refer to the no-wiggle and wiggle parts of the linear / 1-loop power spectrum and :math:`Sigma^2` is the BAO damping scale (computed in the _prepare_IR_resummation function)

        Returns:
            np.ndarray: Vector of IR-resummed linear-plus-one-loop power :math:`P_\mathrm{lin+1,IR}(k)` for the input k-vector.
        """

        if not hasattr(self,'resummed_one_loop_power'):

            # First create IR interpolators if not present
            self._prepare_IR_resummation()

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
                np.ndarray: :math:`W(kR)` evaluated on the input k-vector.
            """
            kR = self.kh_vector*R
            return  3.*(np.sin(kR)-kR*np.cos(kR))/kR**3.

    def _one_loop_only_power_interpolater(self,linear_spectrum):
        """
        Compute the one-loop SPT power interpolator, using the FAST-PT module. This is computed from an input linear power spectrum.

        Note that the FAST-PT output contains large oscillations at high-k. To alleviate this, we perform smoothing interpolation above some k.

        Args:
            linear_spectrum (function): Function taking input wavenumber in h/Mpc units and returning a linear power spectrum.

        Returns:
            scipy.interp1d: An interpolator for the SPT power given an input k (in :math:`h/\mathrm{Mpc}` units).

        """
        if self.verb: print("Computing one-loop power spectrum")
        # Define some k grid for interpolation (with edges well separated from k limits)
        min_k = np.max([np.min(self.kh_vector),1e-4]) # setting minimum to avoid zero errors
        max_k = np.max(self.kh_vector)
        kh_interp = np.logspace(np.log10(min_k)-0.5,np.log10(max_k)+0.5,self.OneLoop_N_k)

        # Compute the one-loop spectrum using FAST-PT
        fastpt = FASTPT.FASTPT(kh_interp,to_do=['one_loop_dd'],n_pad=len(kh_interp)*3,
                               verbose=0);
        initial_power=fastpt.one_loop_dd(linear_spectrum(kh_interp).copy(),C_window=0.65,P_window=[0.25,0.25])[0]

         # Now convolve k if necessary
        filt = kh_interp>self.OneLoop_k_cut
        if np.sum(filt)==0:
            combined_power = initial_power
            combined_k = kh_interp
        else:
            convolved_k = np.convolve(kh_interp[filt],np.ones(self.OneLoop_N_interpolate,)/self.OneLoop_N_interpolate,mode='valid')
            convolved_power = np.convolve(initial_power[filt],np.ones(self.OneLoop_N_interpolate,)/self.OneLoop_N_interpolate,mode='valid')

            # Concatenate to get an output
            combined_power = np.concatenate([initial_power[kh_interp<min(convolved_k)],convolved_power])
            combined_k = np.concatenate([kh_interp[kh_interp<min(convolved_k)],convolved_k])

        # Zero any power values with kh < kh_min
        combined_power[combined_k<self.kh_min] = 0.

        # Create and return an interpolator
        return interp1d(combined_k,combined_power)

    def _prepare_IR_resummation(self):
        """
        Compute relevant quantities to allow IR resummation of the non-linear power spectrum to be performed. This computes the no-wiggle power spectrum, from the 4th order polynomial scheme of Hamann et al. 2010.

        A group of spectra for the no-wiggle linear and no-wiggle 1-loop power are output for later use. The BAO damping scale

        .. math::

            \Sigma^2 =  \frac{1}{6\pi^2}\int_0^\Lambda dq\,P_\mathrm{NL}^{nw}(q)\left[1-j_0(q\ell_\mathrm{BAO})+2j_2(q\ell_\mathrm{BAO})\right]

        is also computed.

        This function is empty if spectra and :math:`Sigma^2` have already been computed.

        """

        if not hasattr(self,'linear_no_wiggle_power') and not hasattr(self,'one_loop_only_no_wiggle_power') and not hasattr(self,'BAO_damping'):

            # First define a k-grid in h/Mpc units
            min_k = np.max([np.min(self.kh_vector),1e-4]) # setting minimum to avoid zero errors
            max_k = np.max(self.kh_vector)
            kh_interp = np.logspace(np.log10(min_k)-0.5,np.log10(max_k)+0.5,self.IR_N_k)

            # Define turning point of power spectrum (we compute no-wiggle spectrum beyond this point)
            linear_power_interp = self.cosmology.compute_linear_power(kh_interp,kh_min=self.kh_min)
            max_pos = np.where(linear_power_interp==max(linear_power_interp))
            kh_turn = kh_interp[max_pos]
            Pk_turn = linear_power_interp[max_pos]
            Pk_max = self.cosmology.compute_linear_power(np.atleast_1d(self.IR_kh_max),kh_min=self.kh_min)

            # Define k in required range
            ffilt = np.where(np.logical_and(kh_interp>kh_turn,kh_interp<self.IR_kh_max))
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
                f2 = logP2 - poly24(np.log(self.IR_kh_max))
                a1 = (f1-f2)/(np.log(kh_turn)-np.log(self.IR_kh_max))
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
            self.one_loop_only_no_wiggle_power = self._one_loop_only_power_interpolater(linear_no_wiggle_interp)(self.kh_vector)

            # Compute the BAO smoothing scale Sigma^2
            def _BAO_integrand(q):
                r_BAO = 105. # BAO scale in Mpc/h
                kh_osc = 1./r_BAO
                return self.cosmology.compute_linear_power(q,kh_min=self.kh_min)*(1.-spherical_jn(0,q/kh_osc)+2.*spherical_jn(2,q/kh_osc))/(6.*np.pi**2.)
            kk_grid = np.linspace(1e-4,0.2,10000)

            # Now store the BAO damping scale as Sigma^2
            self.BAO_damping = simps(_BAO_integrand(kk_grid),kk_grid)
            if self.verb: print('Non-linear BAO damping scale = %.2f Mpc/h'%np.sqrt(self.BAO_damping))
