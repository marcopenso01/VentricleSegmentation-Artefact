import logging
import os
import os.path
import logging
import random

import cv2
import numpy as np
from skimage import exposure

import configuration as config
import image_utils

def augmentation_function(images, labels):
    '''
    :param images: A numpy array of shape [batch, X, Y], normalized between 0-1
    :param labels: A numpy array containing a corresponding label mask     
    ''' 
    # Define in configuration.py which operations to perform
    do_rotation_range = config.do_rotation_range
    do_gamma = config.gamma
    angle = config.rg
    
    # Probability to perform a generic operation
        
    new_images = []
    new_labels = []
    num_images = images.shape[0]

    for i in range(num_images):

        #extract the single image
        img = np.squeeze(images[i,...])
        lbl = np.squeeze(labels[i,...])

        # RANDOM ROTATION
        if do_rotation_range:
            random_angle = np.random.uniform(angle[0], angle[1])
            img = image_utils.rotate_image(img, random_angle)
            lbl = image_utils.rotate_image(lbl, random_angle, interp=cv2.INTER_NEAREST)

        # RANDOM GAMMA CORRECTION
        if do_gamma:
            gamma = random.randrange(8,13,1)
            img = exposure.adjust_gamma(img, gamma/10)

        new_images.append(img[..., np.newaxis])
        new_labels.append(lbl[...])

    sampled_image_batch = np.asarray(new_images)
    sampled_label_batch = np.asarray(new_labels)

    return sampled_image_batch, sampled_label_batch
