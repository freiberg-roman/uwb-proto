class BaseGenerator:
    """Base class for generators.

    Generators should provide of synthesis measurements for a location. For performance reasons
    generation of data should be initialized by the :meth:`uwb.generator.BaseGenerator.gen`. and
    :meth:`uwb.generator.BaseGenerator.get_closest_position` should provide the indices for the
    given coordinates.
    """

    def __init__(self):
        pass

    def gen(self):
        """Initializes the generation process."""
        pass

    def __iter__(self):
        """Iterator implementation for generator.

        Returns the data per measurement location. General layout should be
        (np.array, index, position), where the position should match the shape of the underlying
        structure.
        """
        pass

    def get_closest_position(self, coordinates):
        """Finds indices in the map for the given coordinates.

        Method provides indices in the map for the underlying geometry of the location.

        Attributes:
            coordinates: coordinates to look up map position.
        """
        pass

    @property
    def shape(self):
        """Shape of the underlying structure.

        Shape of the underlying structure. This will be used for pre-allocation. In most cases
        a grid structure is preferable. Having positions with no measurements are easier to deal
        with than obscure shapes. For memory efficiency this decision can be altered.
        """
        return (1,)
