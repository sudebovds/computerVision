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
earthE = load_image(EARTH_EAST)
earthW = load_image(EARTH_WEST)

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

mars_pix = pixelate_image(mrs, (3, 3), (300, 300))
earthE_pix = pixelate_image(earthE, (3, 3), (300, 300))
earthW_pix = pixelate_image(earthW, (3, 3), (300, 300))

def color_pie(image, title):
    """Average an image's RGB channels and plot the results as a pie chart."""
    r, g, b = cv2.split(image)
    color_aves = []
  
    for array in (r, g, b):
        color_aves.append(np.average(array))

    labels = 'Red', 'Green', 'Blue'
    colors = ['red', 'green', 'blue'] 
    fig, ax = plt.subplots(figsize=(3.5, 3.3)) # size in inches
    _, _, autotexts = ax.pie(color_aves,
                            labels=labels,
                            autopct='%1.1f%%',
                            colors=colors)
    for autotext in autotexts:
        autotext.set_color('white')
    plt.title(f'{title}\n')

    plt.show()

color_values_mars = mars_pix[150, 150]
color_values_earthE = earthE_pix[150, 150]
color_values_earthW = earthW_pix[150, 150]

print(f'Mars colors = {color_values_mars} \n')
print(f'Earth East colors = {color_values_earthE} \n')
print(f'Earth West colors = {color_values_earthW} \n')

color_pie(mars_pix, 'Mars')
color_pie(earthE_pix, 'Earth East')
color_pie(earthW_pix, 'Earth West')