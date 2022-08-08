from pydantic import validator
from ..BaseObject import BaseColouredObject


class BaseSolidObject(BaseColouredObject):
    """
    Base class for all solid objects
    Keywords:
        shine:
            the shininess of the where (following the Blinn-Phong model). Scalar value must be
            between 0 and 100. The more shiny a sphere is, the brighter it reacts to light
        reflect:
            The reflectivity of the object (not be be confused with shine). Scalar value between 0 and 1.
            The more reflective the sphere is, the easier it is to see other objects reflected by this sphere
        Plus the other keywords from BaseColourdObject class, namely "name", "coordinates", "ambient", "diffuse" and
        "specular"
    """
    shine: float
    reflect: float

    @validator('shine')
    def _validate_shine(cls, shine):
        """
        Ensures shine is between 0 and 100
        """
        if not 0 <= shine <= 100:
            raise ValueError(f'Shine must be between 0 and 100')
        return shine

    @validator('reflect')
    def _validate_reflect(cls, reflect):
        """
        Ensures reflect is between 0 and 1
        """
        if not 0 <= reflect <= 1:
            raise ValueError(f'Reflect must be between 0 and 1')
        return reflect
