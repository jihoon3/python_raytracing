from .BaseSolidObject import BaseSolidObject
from pydantic import validator
import numpy as np


class Sphere(BaseSolidObject):
    """
    Class for spheres
    Keywords:
        coordinates:
            the location of the centre of the sphere (np array shape (3,))
        name:
            the name of the sphere. Must be unique (i.e no two spheres may share the same name)
        ambient:
            the ambient of the sphere (following the Blinn-Phong model). Values must be between 0 and 1.
            Np array shape (3,)
        diffuse:
            the diffuse of the sphere (following the Blinn-Phong model). Values must be between 0 and 1.
            Np array shape (3,)
        specular:
            the specular of the sphere (following the Blinn-Phong model). Values must be between 0 and 1.
            Np array shape (3,)
        shine:
            the shininess of the where (following the Blinn-Phong model). Scalar value must be
            between 0 and 100. The more shiny a sphere is, the brighter it reacts to light
        reflect:
            The reflectivity of the object (not be be confused with shine). Scalar value between 0 and 1.
            The more reflective the sphere is, the easier it is to see other objects reflected by this sphere
        radius:
            the radius of sphere. Scalar value must be positive

    """
    radius: float  # radius

    @validator('radius')
    def _validate_radius(cls, radius):
        """
        Checks the positive-ness of radius
        """
        if radius <= 0:
            raise ValueError(f'Radius must be a positive value')
        return radius

    @classmethod
    def create_flat_surface(
            cls,
            name: str,
            north: np.ndarray,
            east: np.ndarray,
            point_on_surface: np.ndarray,
            ambient: np.ndarray,
            diffuse: np.ndarray,
            specular: np.ndarray,
            radius: int = 100000,
            shine: int = 45,
            reflect: float = 0.1,
    ) -> 'Sphere':
        """
        Define a plane by defining north and east, as well a point on the plane.
        By setting radius to a large number, this method will create a sphere that will
        look flat in the image (a bit like earth - it appears flat as the radius is large)

        The centre of the sphere is calculated by taking the cross between north and east, and going
        <radius> units in that direction (from the reference point)
        """
        centre_direction = np.cross(north, east)
        centre_direction = (centre_direction / np.linalg.norm(centre_direction)).astype('float32')
        centre = (point_on_surface + centre_direction * radius).astype('float32')
        return Sphere(
            name=name,
            coordinates=centre,
            radius=radius,
            ambient=ambient,
            diffuse=diffuse,
            specular=specular,
            shine=shine,
            reflect=reflect
        )
