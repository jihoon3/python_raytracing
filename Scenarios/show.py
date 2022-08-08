from PIL import Image
import numpy as np
import threading


def show_image(image: np.ndarray, show: bool = False):
    """
    Function displays the given image if show is true
    Args:
        image:
            the numpy array of the image (in rgb)
        show:
            if False, function will not show
    """
    if show:
        threading.Thread(target=lambda: Image.fromarray((image * 255).astype('uint8')).show()).start()
