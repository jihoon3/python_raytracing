import numpy as np
from SceneInterface import scene


class _AutoNumpyUpdate(np.ndarray):
    """
    This class replaces the numpy arrays defined in the dataclasses.
    This is to ensure the modification of any values results in _SceneInterface object scene being aware
    of the changes
    """
    def __new__(cls, *args, _linked_dataclass=None, **kwargs):
        obj = np.array(*args, **kwargs).view(cls)
        return obj

    def __init__(self, *args, _linked_dataclass=None, **kwargs):
        if _linked_dataclass is None:
            raise TypeError(f'{self.__class__.__name__} initialisation error')
        super().__init__()
        self._linked_dataclass = _linked_dataclass

    @staticmethod
    def __array_finalize__(viewed):
        if viewed is None:
            return

    def __setitem__(self, name, value):
        super().__setitem__(name, value)
        scene._assign_updated(self._linked_dataclass)
