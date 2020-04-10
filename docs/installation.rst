Installation
============

EffectiveHalos can be installed either from pip or by cloning the GitHub repository. Make sure to install the :ref:`dependencies` first!


Installation via pip
---------------------

We recommend that EffectiveHalos is installed via pip::

  pip install EffectiveHalos (--user)

This installs the latest release of the code.


Installation from source
-------------------------

EffectiveHalos can also be installed directly from the git repository::

  git clone https://github.com/EffectiveHalos.git
  cd EffectiveHalos
  python -m pip install . (--user)

This will install the current master branch of the git repository.

.. _dependencies:

Dependencies
--------------

**Basic Dependencies**:

- numpy
- scipy
- cython (for CLASS)

**CLASS**

To run EffectiveHalos, we require the Boltzmann code CLASS along with its Python wrapper classy. This can be installed from the CLASS `Github <https://github.com/lesgourg/class_public>`_ and is used to compute the linear power spectrum for a specified cosmology.

The basic installation follows::

  # Clone the class repository
  git clone https://github.com/lesgourg/class_public.git

  # Now install
  cd class_public
  make clean
  make

For further details, including the modifications required for Mac compilation, see the CLASS `wiki <https://github.com/lesgourg/class_public/wiki/Installation>`_. Note that, if a modified version of CLASS is installed (e.g. CLASS-PT) which modifies the classy wrapper, EffectiveHalos will use this instead.

**FAST-PT**

EffectiveHalos uses Joe McEwen's `FAST-PT <https://github.com/JoeMcEwen/FAST-PT>`_ package to compute one-loop power spectra from the CLASS linear power spectrum. It is easiest to install this from pip::

  pip install fast-pt (--user)

**mcfit**

EffectiveHalos uses the `mcfit <https://github.com/eelregit/mcfit>`_ for integral transforms. It is easiest to install this from pip::

  pip install mcfit (--user)
