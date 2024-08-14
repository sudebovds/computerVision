import requests
import numpy as np
import cv2
from matplotlib import pyplot as plt

EARTH_EAST = 'https://bit.ly/3SGPzCO'
EARTH_WEST = 'https://go.nasa.gov/3Wz74WQ'
MARS = 'https://go.nasa.gov/3Wuf6Qs'

def load_image(url):
    response = requests.get(url)
    image_array = np.asarray(bytearray(response.content), dtype="uint8")
    img_ini = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img_ini, cv2.COLOR_BGR2RGB)

mrs = load_image(MARS)

def pixelate_image(image, target_matrix, viz_matrix):
    """
    Load and pixelate an image, then enlarge it for visualization.

    Arguments:
        image: name of the image (as RGB NumPy array)
        target_matrix: tuple of pixelation size (such as (3, 3) for 3x3) 
        viz_matrix: tuple of final size (such as (300, 300) for 300x300)
   
    Returns:
        NumPy ndarray        
    """
    # Pixelate, resize, and return the image:
    pixelated = cv2.resize(image, target_matrix, interpolation=cv2.INTER_AREA)
    return cv2.resize(pixelated, viz_matrix, interpolation=cv2.INTER_NEAREST)

mars = pixelate_image(mrs, (3, 3), (300, 300))

plt.imshow(mars)
plt.axis('off')
plt.show()