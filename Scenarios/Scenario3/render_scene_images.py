import Scenarios.Scenario3.solid_objects
import Scenarios.Scenario3.meta_objects
# Need to import above to population the scene with the necessary objects

from SceneInterface import scene
from Scenarios import show_image, save_image
import numpy as np
import os

_savedir = f'output_media{os.sep}Scenario3'


def render_scene_images(save: bool = False, show: bool = True) -> np.ndarray:
    """
    Function runs scenario 3 (3 frames)
    Args:
        save:
            boolean. If true, function will save all outputs as a .png file to the directory output_media/scenario3
            with the appropriate names
        show:
            boolean. If true, function will display the outputs (without waiting in between) using the PILLOW module
    Returns:
        a numpy array of size (3, 800, 1280, 3) corresponding to 3 frames with resolution 800 x 1280 with
        3 channels (rgb)
    """
    # Each section is a separate frame. Only the light source and camera details are changed each time
    # r1 -> 1 reflection
    scene.set_reflect(reflect=1)
    frame = scene.capture_frame()
    show_image(image=frame, show=show)
    save_image(save_dir=_savedir, image=frame, image_name='r1', save=save)

    # r3
    scene.set_reflect(reflect=3)
    frame = scene.capture_frame()
    show_image(image=frame, show=show)
    save_image(save_dir=_savedir, image=frame, image_name='r3', save=save)

    # r5
    scene.set_reflect(reflect=5)
    frame = scene.capture_frame()
    show_image(image=frame, show=show)
    save_image(save_dir=_savedir, image=frame, image_name='r5', save=save)

    return scene.frames
