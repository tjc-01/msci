"""Provides the CachedRandomGenerator class."""
import numpy as np


class CachedRandomGenerator:
    """A CachedRandomGenerator acts as a numpy Random Generator that can cache values.

    This class has all the same features as a numpy Random Generator but can also cache
    the last object generated.
    """

    def __init__(self, seed=None):
        """Instantiate a new CachedRandomGenerator.

        Args:
            seed (int): Some seed value for the generator.

        Returns:
            srg (CachedRandomGenerator): The CachedRandomGenerator object.
        """
        self.__generator = np.random.default_rng(seed)
        self.__cache = None

    def generate(self, func: str, *args, **kwargs):
        """Generates random output from the Random Generator.

        Args:
            func (str): A string x where x is some np.random.Generator.x corresponding
                to the desired generator method.
            *args: Args for the method denoted by func.
            **kwargs: Kwargs for the method denoted by func.

        Returns:
            output: The results of the method denoted by func.

        """
        res = getattr(self.__generator, func)(*args, **kwargs)
        self.__cache = res
        return res

    def get_cache(self):
        """Gets the cached output (i.e. whatever was last generated).

        Returns:
            cache: The results from the last call of CachedRandomGenerator.generate.

        """
        return self.__cache
