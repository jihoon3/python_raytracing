from ..BaseObject import BaseObject
from pydantic import validator, Field
import numpy as np
from typing import Tuple


class Camera(BaseObject):
    """
    Class for Camera. This is a unique class (one instance only).
    Keywords:
        coordinates:
            the coordinates of the viewer (np array shape (3,))
        name:
            must be "_camera"
        resolution:
            pair of integers representing (height, width)
        background_colour:
            rgb with values between 0 and 1 (np array shape (3,))
        screen_vectors:
            a pair of vectors cam_to_screen, screen_north - each being np arrays of shape (3,)
            cam_to_screen represents the vector from the camera to the screen (class will normalise it for you),
            and cam_north is the direction of screen north (class will normalise it for you).
            Both must be not the zero vector, the two vectors must be orthogonal
        Note rays will be initialised automatically, and if provided the constructor will ignore
        the provided value
    """
    name: str = Field('_camera', allow_mutation=False)
    resolution: Tuple[int, int] = Field(..., allow_mutation=False)
    background_colour: np.ndarray
    screen_vectors: Tuple[np.ndarray, np.ndarray]

    @validator('resolution')
    def _validate_resolution(cls, resolution):
        """
        Checks that resolution = (h, w) is valid
        """
        if resolution[0] <= 0 or resolution[1] <= 0:
            raise ValueError(f'Resolution must be positive integers')
        return resolution

    @validator(
        'screen_vectors'
    )
    def _validate_screen_vecs(
            cls,
            screen_vectors: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validates that cam_to_screen and screen_north are valid orthogonal vectors
        """
        cam_to_screen, screen_north = screen_vectors
        cls._check_coordinates(cam_to_screen)
        cls._check_coordinates(screen_north)
        if all(i == 0 for i in cam_to_screen):
            raise ValueError(f'Given cam_to_screen is the zero vector')
        if all(i == 0 for i in screen_north):
            raise ValueError(f'Given screen_north is the zero vector')

        cam_to_screen = cam_to_screen / np.linalg.norm(cam_to_screen)
        screen_north = screen_north / np.linalg.norm(screen_north)

        if abs(np.dot(cam_to_screen, screen_north)) >= 0.000005:
            raise ValueError(f'cam_to_screen and screen_north must be orthogonal vectors')

        return cam_to_screen, screen_north

    def _construct_rays(self) -> np.ndarray:
        """
        Initialises the rays (unit vector of each ray direction (towards each pixel)
        originating from the camera)
        """
        resolution = self.resolution
        cam_position = self.coordinates
        screen_vectors = self.screen_vectors
        cam_position = np.array(cam_position, dtype='float32')
        cam_to_screen = np.array(screen_vectors[0], dtype='float32')
        screen_north = np.array(screen_vectors[1], dtype='float32')
        screen_centre = cam_position + cam_to_screen
        screen_east = np.cross(cam_to_screen, screen_north)
        screen_locs = np.array(
            [
                np.linspace(
                    (
                            screen_centre +
                            0.5 * screen_east * (1 / resolution[1] - 1) +
                            0.5 * screen_north * ((resolution[0] - 1 - 2 * i) / resolution[1])
                    ),
                    (
                            screen_centre +
                            0.5 * screen_east * (-1 / resolution[1] + 1) +
                            0.5 * screen_north * ((resolution[0] - 1 - 2 * i) / resolution[1])
                    ),
                    resolution[1],
                    dtype='float32'
                )
                for i in range(resolution[0])
            ],
            dtype='float32'
        )
        rays = screen_locs - cam_position
        return rays / np.linalg.norm(rays, axis=2).reshape((rays.shape[0], rays.shape[1], 1))
