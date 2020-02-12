Installation
============

.. todo::

  finish this


Dependencies
--------------

**CLASS**

To run PerturbedHalos, we require the Boltzmann code CLASS along with its Python wrapper classy. This can be installed from the CLASS `Github <https://github.com/lesgourg/class_public>`_ and is used to compute the linear power spectrum for a specified cosmology.

**FAST-PT**

PerturbedHalos uses Joe McEwen's `FAST-PT <https://github.com/JoeMcEwen/FAST-PT>`_ package to compute one-loop power spectra from the CLASS linear power spectrum. It is easiest to install this from pip::

  pip install fast-pt --user

**Other Dependencies**
- numpy
- scipy
- cython (for CLASS)

These are likely already installed, else they can be installed from pip.

.. todo::

  can we get this autoinstalled?

Installing PerturbedHalos
--------------------------

Once the above packages are installed, PerturbedHalos can be simply installed via pip::

  pip install PerturbedHalos --user

To check this is installed run::

.. todo::

  write a test script
