import numpy as np
def PIL2array(img):
    return np.array(img.getdata(),np.uint8).reshape(img.size[1], img.size[0], 3)
