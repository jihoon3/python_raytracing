import numpy as np
import os
import threading
import cv2


def save_image(save_dir: str, image: np.ndarray, image_name: str, save: bool = False):
    """
    Function saves the image if save is true,
    Args:
        save_dir:
            the path to the repository to save image to
        image_name:
            the name of the image file (without the extension, function will automatically save as a .png)
        image:
            the numpy array of the image (in rgb)
        save:
            if False, function will not save
    """
    if save:
        os.makedirs(save_dir, exist_ok=True)
        threading.Thread(
            target=lambda: cv2.imwrite(
                f'{save_dir}{os.sep}{image_name}.png',
                (image * 255).astype('uint8')[:, :, ::-1]
            )
        ).start()

