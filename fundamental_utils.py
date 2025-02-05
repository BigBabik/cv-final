import cv2 as cv
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

def create_fundamental_matrix(mkpts0, mkpts1):
    F, mask = cv.findFundamentalMat(
        np.array(mkpts0),
        np.array(mkpts1),
        cv.USAC_MAGSAC,
        ransacReprojThreshold=1.25,
        confidence=0.99999,
        maxIters=10000)

    if F is not None:
        if F.shape != (3,3):
            print(F.shape)
        F = F.reshape(-1)
        F = F[:9]

    return F, mask


def ComputeMaa(err_q, err_t):
    '''Compute the mean Average Accuracy at different tresholds, for one scene.'''
    thresholds_q = np.linspace(1, 10, 10)
    thresholds_t = np.geomspace(0.2, 5, 10)
    assert len(err_q) == len(err_t)
    
    acc, acc_q, acc_t = [], [], []
    for th_q, th_t in zip(thresholds_q, thresholds_t):
        acc += [(np.bitwise_and(np.array(err_q) < th_q, np.array(err_t) < th_t)).sum() / len(err_q)]
        acc_q += [(np.array(err_q) < th_q).sum() / len(err_q)]
        acc_t += [(np.array(err_t) < th_t).sum() / len(err_t)]
    return np.mean(acc), np.array(acc), np.array(acc_q), np.array(acc_t)

def compare_matrixes(F1, F2):
    pass


def predict_for_pair(npz_file, directory_path):
    # Load the npz file
    data = np.load(os.path.join(directory_path, npz_file))
    exception = None

    # Extract matches and keypoints
    matches = data['matches']
    kpts0 = data['keypoints0']
    kpts1 = data['keypoints1']
    match_conf = data['match_confidence']
    
    # Filter valid matches
    valid = []
    cnt = 0
    bad = 0
    mkpts0 = []
    mkpts1 = []
    THRESH = 0.7

    AMOUNT = 60
    THRESH = 0.95
    while cnt < AMOUNT and THRESH >= 0.1:
        mkpts0 = []
        mkpts1 = []
        cnt = 0
        for i in range(len(matches)):
            is_valid = matches[i] > -1 and match_conf[i] >= THRESH
            if is_valid:
                cnt += 1
                mkpts0.append(kpts0[i])
                mkpts1.append(kpts1[matches[i]])
        THRESH -= 0.05
    

    mkpts0 = np.array(mkpts0)
    mkpts1 = np.array(mkpts1)
    
    #valid = (matches) >= 0 & (match_conf >= THRESH)
    #mkpts0 = kpts0[valid]
    #mkpts1 = kpts1[matches[valid]]
    retry = 0
    passed = False
    while retry < 4 and not passed:
        try:
        # Estimate the fundamental matrix using the matched keypoints
            #F, mask = cv2.findFundamentalMat(np.array(mkpts0), np.array(mkpts1), cv2.FM_RANSAC, confidence=0.999999)
            F, mask = cv.findFundamentalMat(
                mkpts0,
                mkpts1, 
                cv.USAC_MAGSAC, 
                ransacReprojThreshold=0.1, # was 0.5
                confidence=0.3,
                maxIters=10000)
            passed = True
        except Exception as e:
            F = None
            exception = type(e).__name__
            if len(mkpts0) <= 8:
                print(f"Found {len(mkpts0)}")
                retry = 4
                continue
            print(e)
            F = None
            retry += 1
            print(f"Retry: {retry}")
    
    
    # Extract scene name and image names
    scene_name = data['scene_name']
    image_name0 = npz_file.split('-')[0]
    image_name1 = npz_file.split('-')[1]
    
    
    # Append the results to the lists
    sample = f"{scene_name};{image_name0}-{image_name1}"
    
    if F is not None:
        if F.shape != (3,3):
            print(F.shape)
        F = F.reshape(-1)
        F = F[:9]
        
    return sample, F, exception


def get_F_dataframe(directory):

    # List all files in the directory
    files = os.listdir(directory)

    # Filter out the npz files
    npz_files = [file for file in files if file.endswith('.npz')]
    samples = []
    fundamental_matrices = []
    exception_counts = {}


    # Iterate over all npz files
    for npz_file in tqdm(npz_files):
        sample, F, exp = predict_for_pair(npz_file, directory)
        samples.append(sample)
        fundamental_matrices.append(F)
        if exp in exception_counts:
                exception_counts[exp] += 1
        else:
            exception_counts[exp] = 1
            print(exp)
    total_errors = sum(exception_counts.values())
    print(F"total malformed {total_errors}")
    # Create a dataframe with the collected data
    df = pd.DataFrame({
        'sample_id': samples,
        'fundamental_matrix': fundamental_matrices
    })

    return df

    