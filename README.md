# EffectiveHalos
EffectiveHalos is a fast Python code providing models of the real-space matter power spectrum, based a combination of the Halo Model and Effective Field Theory, which are 1\% accurate up to k = 1 h/Mpc, across a range of cosmologies, including those with massive neutrinos. It can additionally compute accurate halo count covariances (including a model of halo exclusion), both alone and in combination with the matter power spectrum.

This is based on the work of [Philcox, Spergel \& Villaescusa-Navarro (2020)](https://arxiv.org/abs/2004.09515), and full documentation is availble on [ReadTheDocs](https://EffectiveHalos.rtfd.io).

## Authors
**Main Authors**
- Oliver Philcox (Princeton)

**Collaborators**
- David Spergel (Princeton / CCA)
- Francisco Villaescusa-Navarro (Princeton / CCA)

## Installation

EffectiveHalos can be simply installed using pip:

```
pip install EffectiveHalos (--user)
```

Note that you will need a [CLASS](https://github.com/lesgourg/class_public) installation, including the 'classy' Python wrapper, to run EffectiveHalos.


## Basic Usage

To compute a matter power spectrum in EffectiveHalos, simply run the following:

```python
from EffectiveHalos import *
import numpy as np

## Parameters
z = 0. # redshift
cs2 = 8. # effective speed of sound (should be calibrated from simulations)
R = 1. # smoothing scale (should be calibrated from simulations)
k = np.arange(0.01, 1., 0.005) # wavenumbers in h/Mpc

## Load general classes
cosmology = Cosmology(z, 'Planck18') # use Planck 2018 cosmology
mass_function = MassFunction(cosmology, 'Bhattacharya') # Bhattacharya 2010 mass function
halo_physics = HaloPhysics(cosmology, mass_function, 'Duffy', 'NFW') # Duffy 08 concentration relation, NFW halo profiles

## Load HaloModel class
halo_model = HaloModel(cosmology, mass_function, halo_physics, k)

## Compute the power spectrum in both Effective and Standard Halo Models
power_spectrum_EHM = halo_model.halo_model(cs2, R)
power_spectrum_SHM = halo_model.halo_model(cs2, R, 'Linear', 0, 0, 0)
```

This generates an estimate for the matter power spectrum in a few seconds. Let's plot this:

![alt text](https://github.com/oliverphilcox/EffectiveHalos/blob/master/docs/ehm_tutorial_spec.png "Effective Halo Model Power Spectrum")

A full tutorial can be found [here](https://effectivehalos.readthedocs.io/en/latest/Tutorial.html).

***New for version 1.1:*** Accurate models for the power spectrum in massive neutrino cosmologies.
