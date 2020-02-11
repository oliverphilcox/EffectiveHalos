"""
Created on 29 July 2012
@author: Lisa Simpson
"""

class DatabaseManager(object):
    """
    Create and manage a new sqlite database.

    Methods:
        __init__: Initialize the class at a given redshift with a specified cosmology.
        linear_power: Compute the linear power spectrum for a given k vector.
        sigmaM: Compute sigma(M) from CLASS for input mass M.
        sigma_logM_int: Return an interpolated value of sigma(M) corresponding to an input log10(M) grid.
        dlns_dlogM_int: Return an interpolated value of d(ln(sigma(M)))/d(log10(M)) corresponding to an input log10(M) grid.
    """

    def method(self,index):
        """
        A simple method.

        Args:
            input (int): an input

        Returns:
            1
        """
        return 1;
