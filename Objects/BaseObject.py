from pydantic import BaseModel, validator, Field
from typing import ClassVar, Any
from SceneInterface import scene
import numpy as np
from ._AutoNumpyUpdate import _AutoNumpyUpdate


class BaseObject(BaseModel):
    """
    Base class for all physical and meta objects in the scene
    Keywords:
        name:
            the name of the object. Note that all objects must have a unique name, and that the camera must
            have the name "_camera", and the light must have the name "_light"
        coordinates:
            the positioning of the object. Must be a np array of shape (3,)
        description:
            An optional descriptor string. Has no functionality, but is provided for convenience
    """
    name: str = Field(..., allow_mutation=False)  # unique name
    coordinates: np.ndarray
    description: str = None

    def __init__(self, **kwargs):
        for forbidden in {'screen_array', 'rays', 'object_class_type'}:
            if forbidden in kwargs:
                del kwargs[forbidden]
        super().__init__(**kwargs)
        scene.register_object(self)
        self.Config.validate_assignment = False
        for name, value in self.dict().items():
            if isinstance(value, np.ndarray):
                super().__setattr__(
                    name,
                    _AutoNumpyUpdate(value.copy(), _linked_dataclass=self)
                )
            if isinstance(value, (1,).__class__):
                if any(isinstance(i, np.ndarray) for i in value):
                    new_tuple_list = []
                    for i in value:
                        if isinstance(i, np.ndarray):
                            new_tuple_list.append(_AutoNumpyUpdate(i.copy(), _linked_dataclass=self))
                        else:
                            new_tuple_list.append(i)
                    super().__setattr__(name, tuple(new_tuple_list))
        self.Config.validate_assignment = True

    def __setattr__(self, key: str, value: Any):
        """
        Automatically sets scene attributes such as scene._camera_updated to True
        """
        super().__setattr__(key, value)
        if key.startswith('_'):
            return
        scene._assign_updated(self)

    @validator('name')
    def _validate_name(cls, name: str):
        """
        Ensures that the name is valid according to _SceneInterface._check_name
        """
        scene._check_name(name, cls)
        return name

    @validator(
        'coordinates',
        check_fields=False
    )
    def _validate_vector(cls, vector: np.ndarray):
        """
        Ensures that the given item is a valid vector
        """
        cls._check_coordinates(vector)
        return vector

    @validator(
        'ambient',
        'specular',
        'diffuse',
        'background_colour',
        check_fields=False
    )
    def _validate_colour_vector(cls, vector: np.ndarray):
        """
        Ensures that all values are between 0 and 1
        """
        cls._check_coordinates(vector)
        if not all(0 <= i <= 1 for i in vector):
            raise ValueError(f'Given colour values must be between 0 and 1')
        return vector

    @staticmethod
    def _check_coordinates(vector: np.ndarray) -> None:
        """
        Ensures vector has shape (3,) and has dtype float32
        """
        if vector.dtype != 'float32':
            raise TypeError(f'The dtype of given vector(s) must be float32')
        if vector.shape != (3,):
            raise ValueError(f'The shape of given vector(s) must be (3,)')

    def de_register(self):
        """
        Delete the object
        Ensure object is de-registered from scene first
        """
        scene.de_register_object(self.name)
        del self

    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True


class BaseColouredObject(BaseObject):
    """
    Base class for all objects with the Blinn-Phong attributes (shine is omitted as light does not have one)
    Keywords:
        ambient:
            the ambient of the sphere (following the Blinn-Phong model). Values must be between 0 and 1.
            Np array shape (3,)
        diffuse:
            the diffuse of the sphere (following the Blinn-Phong model). Values must be between 0 and 1.
            Np array shape (3,)
        specular:
            the specular of the sphere (following the Blinn-Phong model). Values must be between 0 and 1.
            Np array shape (3,)
        Plus the other keywords from the BaseObject class, namely "name" and "coordinates"
    """
    ambient: np.ndarray
    specular: np.ndarray
    diffuse: np.ndarray
