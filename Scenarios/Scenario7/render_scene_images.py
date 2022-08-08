import Scenarios.Scenario7.solid_objects
import Scenarios.Scenario7.meta_objects
# Need to import above to population the scene with the necessary objects

from SceneInterface import scene
from Scenarios import show_image, save_image
import numpy as np
import os

_savedir = f'output_media{os.sep}Scenario7'


def render_scene_images(save: bool = False, show: bool = True) -> np.ndarray:
    """
    Function runs scenario 7 (8 frames)
    Args:
        save:
            boolean. If true, function will save all outputs as a .png file to the directory output_media/scenario7
            with the appropriate names
        show:
            boolean. If true, function will display the outputs (without waiting in between) using the PILLOW module
    Returns:
        a numpy array of size (8, 1080, 1920, 3) corresponding to 8 frames with resolution 1080 x 1920 with
        3 channels (rgb)
    """
    # Each section is a separate frame. Only the light source and camera details are changed each time
    # 7a
    frame = scene.capture_frame()
    show_image(image=frame, show=show)
    save_image(save_dir=_savedir, image=frame, image_name='7a', save=save)

    # 7b
    scene['_light'].coordinates = np.array([12, -25, 18], dtype='float32')
    frame = scene.capture_frame()
    show_image(image=frame, show=show)
    save_image(save_dir=_savedir, image=frame, image_name='7b', save=save)

    # 7c
    scene['_light'].coordinates = np.array([0, 20, 12], dtype='float32')
    scene['_camera'].coordinates[1] = -55
    viewing_direction = np.array([-8.,  5.,  0.], dtype='float32')
    scene['_camera'].coordinates = scene['_camera'].coordinates + np.array([-8.,  5.,  0.], dtype='float32') * 0.55
    scene['_camera'].screen_vectors = (viewing_direction, scene['_camera'].screen_vectors[1])
    frame = scene.capture_frame()
    show_image(image=frame, show=show)
    save_image(save_dir=_savedir, image=frame, image_name='7c', save=save)

    # 7d
    scene['_light'].coordinates = np.array([0, -15, 12], dtype='float32')
    cam_pos = np.array([-6, -60, 0], dtype='float32')
    viewing_direction = np.array([-8, -45, 0], dtype='float32') - cam_pos
    scene['_camera'].coordinates = cam_pos + viewing_direction * 0.39
    scene['_camera'].screen_vectors = (viewing_direction, scene['_camera'].screen_vectors[1])
    frame = scene.capture_frame()
    show_image(image=frame, show=show)
    save_image(save_dir=_savedir, image=frame, image_name='7d', save=save)

    # 7e
    scene['_light'].coordinates = np.array([-12, 10, 22], dtype='float32')
    cam_pos = np.array([0, -10, 4], dtype='float32')
    viewing_direction = np.array([0, 1, -0.12], dtype='float32')
    screen_north = np.array([0, 0.12, 1], dtype='float32')
    scene['_camera'].coordinates = cam_pos
    scene['_camera'].screen_vectors = (viewing_direction, screen_north)
    frame = scene.capture_frame()
    show_image(image=frame, show=show)
    save_image(save_dir=_savedir, image=frame, image_name='7e', save=save)

    # 7f
    cam_pos = np.array([-3, -8, 0.2], dtype='float32')
    viewing_direction = np.array([-2.5, 17.0, 0.5], dtype='float32') - cam_pos
    cam_pos = cam_pos + viewing_direction * 0.5
    screen_north = np.array([0, -0.012, 1], dtype='float32')
    scene['_camera'].coordinates = cam_pos
    scene['_camera'].screen_vectors = (viewing_direction, screen_north)
    frame = scene.capture_frame()
    show_image(image=frame, show=show)
    save_image(save_dir=_savedir, image=frame, image_name='7f', save=save)

    # 7g
    scene['_light'].coordinates = np.array([12, 10, 22], dtype='float32')
    cam_pos = np.array([8, 1, 0.2], dtype='float32')
    viewing_direction = np.array([-2.5, 17.0, 0.5], dtype='float32') - cam_pos
    cam_pos = cam_pos + viewing_direction * 0.25
    screen_north = np.array([0, - (viewing_direction[2]/viewing_direction[1]), 1], dtype='float32')
    scene['_camera'].coordinates = cam_pos
    scene['_camera'].screen_vectors = (viewing_direction, screen_north)
    frame = scene.capture_frame()
    show_image(image=frame, show=show)
    save_image(save_dir=_savedir, image=frame, image_name='7g', save=save)

    # 7h
    scene['_light'].coordinates = np.array([0, 0, 15], dtype='float32')
    cam_pos = np.array([25, -80, 15], dtype='float32')
    viewing_direction = np.array([0, -20, 0], dtype='float32') - cam_pos
    screen_north = np.array([-0.2, 0.16666666666666666, 1], dtype='float32')
    scene['_camera'].coordinates = cam_pos
    scene['_camera'].screen_vectors = (viewing_direction, screen_north)
    frame = scene.capture_frame()
    show_image(image=frame, show=show)
    save_image(save_dir=_savedir, image=frame, image_name='7h', save=save)

    return scene.frames
