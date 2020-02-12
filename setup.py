import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
     name='PerturbedHalos',
     version='1.0',
     author="Oliver Philcox",
     author_email="ohep2@alumni.cam.ac.uk",
     description="Combining the Halo Model and Perturbation Theory: A 1% Accurate Model to k = 1 h/Mpc",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://perturbedhalos.rtfd.io",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 2",
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
