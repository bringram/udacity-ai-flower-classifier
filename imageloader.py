#
# PROGRAMMER:       Brandon Ingram
# DATE CREATED:     Friday, February 14, 2020
# REVISED DATE:     
#

import numpy as np
import torchvision.transforms.functional as F


def process_image(image):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns a NumPy array.
    
    Parameters:
        image - The PIL image to process
    Returns:
        A NumPy array of the processed image
    """
    # Resize the image identical to model training
    image = F.resize(image, 256)
    width, height = image.size
    
    # Crop the image to the correct dimensions
    left = (width - 224) / 2
    top = (height - 224) / 2
    image = F.crop(image, top, left, 224, 224)
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    image = F.to_tensor(image)
    image = F.normalize(image, mean, std)
    
    return image
