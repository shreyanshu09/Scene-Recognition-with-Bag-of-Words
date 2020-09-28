from PIL import Image

import pdb
import numpy as np

def get_tiny_images(image_paths):

    N = len(image_paths)
    size = 16
    
    tiny_images = []
    
    for each in image_paths:
        image = Image.open(each)
        image = image.resize((size, size))
        image = (image - np.mean(image))/np.std(image)
        image = image.flatten()
        tiny_images.append(image)
        
    tiny_images = np.asarray(tiny_images)

    return tiny_images