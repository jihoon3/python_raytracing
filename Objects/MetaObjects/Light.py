from ..BaseObject import BaseColouredObject
from pydantic import Field


class Light(BaseColouredObject):
    """
    Class for light. The current implementation only supports one instance of light
    Keywords:
        coordinates:
            the location of the centre of the sphere (np array shape (3,))
        name:
            the name of the sphere. Must be unique (i.e no two spheres may share the same name). For
            any Light object, the name must be "_light"
        ambient:
            the ambient of the sphere (following the Blinn-Phong model). Values must be between 0 and 1.
            Np array shape (3,)
        diffuse:
            the diffuse of the sphere (following the Blinn-Phong model). Values must be between 0 and 1.
            Np array shape (3,)
        specular:
            the specular of the sphere (following the Blinn-Phong model). Values must be between 0 and 1.
            Np array shape (3,)
        intensity:
            The intensity of light at a point is inversely proportional to the square of the distance
            I.e intensity_factor = min(k, square_of_distance)/square_of_distance. The square of this intensity
            parameter represents k here.
    """
    name: str = Field('_light', allow_mutation=False)
    intensity: float = 1000
