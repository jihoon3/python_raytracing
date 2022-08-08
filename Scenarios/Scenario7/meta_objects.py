import numpy as np
from Objects.MetaObjects import Camera, Light
# This script contains the light and camera settings for scenario 7

camera_settings = {
        'name': '_camera',
        'coordinates': np.array([0, -70, 0], dtype='float32'),
        'resolution': (1080, 1920),
        'screen_vectors': (
            np.array([0, 1, 0], dtype='float32'),  # cam to screen
            np.array([0, 0, 1], dtype='float32'),  # screen north
        ),
        'background_colour': np.array([0, 0, 0], dtype='float32')
}

light_settings = {
    'name': '_light',
    'coordinates': np.array([0, 20, 12], dtype='float32'),
    'ambient': np.array([1, 1, 1], dtype='float32'),
    'diffuse': np.array([1, 1, 1], dtype='float32'),
    'specular': np.array([1, 1, 1], dtype='float32'),
}


Camera(**camera_settings)
Light(**light_settings)