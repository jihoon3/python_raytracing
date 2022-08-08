import numpy as np
from Objects.SolidObjects import Sphere
# Contains all the data for the spheres involved in scenario 3

Sphere(
    name=f'blue',
    coordinates=np.array([-3.5, 3, 0.6], dtype='float32'),
    ambient=np.array([0.016, 0.076, 0.084], dtype='float32'),
    diffuse=np.array([0.16, 0.76, 0.84], dtype='float32'),
    specular=np.array([0.862, 0.964, 0.98], dtype='float32'),
    radius=1,
    shine=66,
    reflect=0.3
)

Sphere(
    name=f'yellow',
    coordinates=np.array([2, 6, -4], dtype='float32'),
    ambient=np.array([0.1, 0.1, 0], dtype='float32'),
    diffuse=np.array([0.7, 0.7, 0], dtype='float32'),
    specular=np.array([1, 1, 1], dtype='float32'),
    radius=2,
    shine=95,
    reflect=0.95
)

Sphere(
    name=f'red',
    coordinates=np.array([3.5, 4, 0.2], dtype='float32'),
    ambient=np.array([0.1, 0, 0], dtype='float32'),
    diffuse=np.array([0.7, 0, 0], dtype='float32'),
    specular=np.array([1, 1, 1], dtype='float32'),
    radius=0.7,
    shine=45,
    reflect=0.3
)

Sphere(
    name=f'purple',
    coordinates=np.array([0, 10, 1.2], dtype='float32'),
    ambient=np.array([0.125, 0.05000000074505806, 0.125], dtype='float32'),
    diffuse=np.array([0.699999988079071, 0.5, 1.0], dtype='float32'),
    specular=np.array([1, 1, 1], dtype='float32'),
    radius=4,
    shine=45,
    reflect=0.6
)
