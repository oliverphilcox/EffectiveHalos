import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
     name='EffectiveHalos',
     version='1.1.0',
     author="Oliver Philcox",
     author_email="ohep2@cantab.ac.uk",
     description="Combining the Halo Model and Perturbation Theory: A 1% Accurate Model to k = 1 h/Mpc",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://effectivehalos.rtfd.io",
     packages=setuptools.find_packages(),
     install_requires=['numpy', 'scipy', 'matplotlib', 'mcfit', 'fast-pt'],
     classifiers=[
         "Programming Language :: Python :: 2",
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
