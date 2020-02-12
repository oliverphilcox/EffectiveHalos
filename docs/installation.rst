Installation
============

.. todo::

  finish this


Dependencies
--------------

**Basic Dependencies**:

- numpy
- scipy
- cython (for CLASS)

**CLASS**

To run PerturbedHalos, we require the Boltzmann code CLASS along with its Python wrapper classy. This can be installed from the CLASS `Github <https://github.com/lesgourg/class_public>`_ and is used to compute the linear power spectrum for a specified cosmology.

The basic installation follows::

  # Clone the class repository
  git clone https://github.com/lesgourg/class_public.git

  # Now install
  cd class_public
  make clean
  make

For further details, including the modifications required for Mac compilation, see the CLASS `wiki <https://github.com/lesgourg/class_public/wiki/Installation>`_.

**FAST-PT**

PerturbedHalos uses Joe McEwen's `FAST-PT <https://github.com/JoeMcEwen/FAST-PT>`_ package to compute one-loop power spectra from the CLASS linear power spectrum. It is easiest to install this from pip::

  pip install fast-pt --user

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
