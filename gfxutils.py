import math
import numpy as np
import time
import os
import yaml
import logging

logger = logging.getLogger(__name__)


def getTime():
    return time.time()
    #returm time.process_time()
    #return time.clock()


def frameIdentical(image1, image2):
    return image1.shape == image2.shape and not(np.bitwise_xor(image1, image2).any())


def calculateDistance(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist


def imageCopyInto(l_img, s_img, x_offset, y_offset):
    l_img[y_offset:y_offset + s_img.shape[0], x_offset:x_offset + s_img.shape[1], 0] = s_img
    l_img[y_offset:y_offset + s_img.shape[0], x_offset:x_offset + s_img.shape[1], 1] = s_img
    l_img[y_offset:y_offset + s_img.shape[0], x_offset:x_offset + s_img.shape[1], 2] = s_img

    #l_img[100:200, 100:200, 2] = s_img

    # background[offset[0]:offset[0]+img_w       ,offset[1]:offset[1]+img_h] = img
    # background[offset[0]:offset[0]+img.shape[0],offset[1]:offset[1]+img.shape[1]] = img


def readVideoFileConfig(filename):
    config = {
        'crop': None,
        'thresh': 14,
    }

    # check for crop settings for file
    vidYaml = filename + '.yaml'
    if os.path.isfile(vidYaml):
        logger.info("Opening video config file: " + vidYaml)
        with open(vidYaml) as file:
            vidYamlData = yaml.load(file, Loader=yaml.FullLoader)

            crop = []
            if 'x1' in vidYamlData:
                logger.info("  Has Croppings")
                crop.append((vidYamlData['x1'], vidYamlData['y1']))
                crop.append((vidYamlData['x2'], vidYamlData['y2']))
            config['crop'] = crop

            if 'thresh' in vidYamlData:
                logger.info("  Has Tresh")
                config['thresh'] = vidYamlData['thresh']

            return config

    return config
