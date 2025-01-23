import cv2 as cv
import numpy as np

def create_fundamental_matrix(points0, points1):
    F, mask = cv.findFundamentalMat(np.array(points0), np.array(points1), cv.FM_RANSAC)

    if F is not None:
        if F.shape != (3,3):
            print(F.shape)
        F = F.reshape(-1)
        F = F[:9]

    return F