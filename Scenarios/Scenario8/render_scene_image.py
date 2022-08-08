import Scenarios.Scenario8.solid_objects
import Scenarios.Scenario8.meta_objects

# Need to import above to population the scene with the necessary objects

from SceneInterface import scene
from Scenarios import show_image, save_image
import numpy as np
import os

_savedir = f'output_media{os.sep}Scenario8'


def render_scene_images(save: bool = False, show: bool = True) -> np.ndarray:
    """
    Function runs scenario 8 (7 frames)
    Args:
        save:
            boolean. If true, function will save all outputs as a .png file to the directory output_media/scenario8
            with the appropriate names
        show:
            boolean. If true, function will display the outputs (without waiting in between) using the PILLOW module
    Returns:
        a numpy array of size (7, 900, 1600, 3) corresponding to 7 frames with resolution 900 x 1600 with
        3 channels (rgb)
    """
    # Each section is a separate frame

    # Without a shell
    scene.set_reflect(reflect=2)
    shell = scene['purple']
    scene.de_register_object('purple')
    scene['_camera'].coordinates[1] = -9
    scene['_light'].intensity = 100

    frame = scene.capture_frame()
    show_image(image=frame, show=show)
    save_image(save_dir=_savedir, image=frame, image_name='without_outer_shell_overview', save=save)

    # With outer shell (main with 2 reflections)
    scene.register_object(shell)
    scene['_light'].intensity = 3
    frame = scene.capture_frame()
    show_image(image=frame, show=show)
    save_image(save_dir=_savedir, image=frame, image_name='main2', save=save)

    # Reflect set to 4
    scene.set_reflect(reflect=4)
    frame = scene.capture_frame()
    show_image(image=frame, show=show)
    save_image(save_dir=_savedir, image=frame, image_name='main4', save=save)

    # Reflect set to 7
    scene.set_reflect(reflect=7)
    frame = scene.capture_frame()
    show_image(image=frame, show=show)
    save_image(save_dir=_savedir, image=frame, image_name='main7', save=save)

    # focus on blue reflect 2
    scene.set_reflect(reflect=2)
    scene['_camera'].coordinates[1] = 0
    frame = scene.capture_frame()
    show_image(image=frame, show=show)
    save_image(save_dir=_savedir, image=frame, image_name='focus_on_blue2', save=save)

    # focus on blue reflect 4
    scene.set_reflect(reflect=4)
    frame = scene.capture_frame()
    show_image(image=frame, show=show)
    save_image(save_dir=_savedir, image=frame, image_name='focus_on_blue4', save=save)

    # focus on blue reflect 7
    scene.set_reflect(reflect=7)
    frame = scene.capture_frame()
    show_image(image=frame, show=show)
    save_image(save_dir=_savedir, image=frame, image_name='focus_on_blue7', save=save)

    return scene.frames
