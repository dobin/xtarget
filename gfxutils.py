import math
import numpy as np
import time
import os
import yaml
import logging

def getTime():
    return time.time()
    #returm time.process_time()
    #return time.clock()

def frameIdentical(image1, image2):
    return image1.shape == image2.shape and not(np.bitwise_xor(image1,image2).any())

def calculateDistance(x1,y1,x2,y2):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist



def readVideoFileConfig(filename):
    config = {
        'crop': None,
        'thresh': 14,
    }

    # check for crop settings for file
    vidYaml = filename +'.yaml'
    if os.path.isfile(vidYaml):
        logging.info("Opening video config file: " + vidYaml)
        with open(vidYaml) as file:
            vidYamlData = yaml.load(file, Loader=yaml.FullLoader)

            crop = []
            if 'x1' in vidYamlData:
                logging.info("  Has Croppings")
                crop.append((vidYamlData['x1'], vidYamlData['y1']))
                crop.append((vidYamlData['x2'], vidYamlData['y2']))
            config['crop'] = crop

            if 'thresh' in vidYamlData:
                logging.info("  Has Tresh")
                config['thresh'] = vidYamlData['thresh']

            return config

    return config
