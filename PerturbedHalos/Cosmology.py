class Cosmology(object):
    """
    Class to hold the basic cosmology and class attributes.

    This can be initialized by a set of cosmological parameters or a pre-defined name.

    """

    def __init__(self,redshift,name="",**params):

        """
        Initialize the cosmology class with cosmological parameters or a defined model.

        Args:
            redshift (float): Desired redshift
            name (str): Load cosmology from a list of predetermined cosmologies. Currently implemented: Quijote
            **params (**kwargs): Other parameters from CLASS.
        """
        return 0
