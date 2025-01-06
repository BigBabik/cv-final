import numpy as np
from typing import List

def normalize_keypoints(keypoints, K):
    C_x = K[0, 2]
    C_y = K[1, 2]
    f_x = K[0, 0]
    f_y = K[1, 1]
    keypoints = (keypoints - np.array([[C_x, C_y]])) / np.array([[f_x, f_y]])
    return keypoints

def pack_coords(coord_list: List):
    """
    :param coord_list: list of lists of length 2 that are the normalized coords
    """
    return ';'.join([f"({x:.6f}, {y:.6f})" for x, y in coord_list])

def unpack_coords(string_of_coords: str):
    points = string_of_coords.split(';')
    return ([eval(p) for p in points])