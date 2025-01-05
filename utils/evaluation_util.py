import numpy as np
import cv2 as cv
from general_utils import *

# A small epsilon.
eps = 1e-15


def compute_essential_matrix(F, K1, K2, kp1, kp2):
    '''Compute the Essential matrix from the Fundamental matrix, given the calibration matrices. Note that we ask participants to estimate F, i.e., without relying on known intrinsics.'''

    assert F.shape[0] == 3, 'Malformed F?'

    E = np.matmul(np.matmul(K2.T, F), K1).astype(np.float64)
    
    kp1n = normalize_keypoints(kp1, K1)
    kp2n = normalize_keypoints(kp2, K2)
    num_inliers, R, T, mask = cv.recoverPose(E, kp1n, kp2n)

    return E, R, T


def compute_error_for_pair(q_gt, T_gt, q, T, scale):
    '''
    Compute the error metric for a single example.
    
    The function returns two errors, over rotation and translation. These are combined at different thresholds by ComputeMaa in order to compute the mean Average Accuracy.
    '''
    
    q_gt_norm = q_gt / (np.linalg.norm(q_gt) + eps)
    q_norm = q / (np.linalg.norm(q) + eps)

    loss_q = np.maximum(eps, (1.0 - np.sum(q_norm * q_gt_norm)**2))
    err_q = np.arccos(1 - 2 * loss_q)

    # Apply the scaling factor for this scene.
    T_gt_scaled = T_gt * scale
    T_scaled = T * np.linalg.norm(T_gt) * scale / (np.linalg.norm(T) + eps)

    err_t = min(np.linalg.norm(T_gt_scaled - T_scaled), np.linalg.norm(T_gt_scaled + T_scaled))

    return err_q * 180 / np.pi, err_t