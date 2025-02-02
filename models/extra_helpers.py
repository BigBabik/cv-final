# A SUBSET OF UTILS FROM THE REPO AND SOME ADDITIONAL FUNCTIONS

from pathlib import Path
import time
from collections import OrderedDict
from threading import Thread
from models.matching import Matching
from models.utils import *
import numpy as np
import torch
import pandas as pd
import cv2 as cv
import os
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
matplotlib.use('Agg')


torch.set_grad_enabled(False)


# --- ADDITIONS ---

def perform_task(input_csv, input_dir, output_dir, matching_config, resize=[-1, -1], max_length=-1, scene_exclude = []):

    input_csv = "/Users/yoav/Documents/Yoav/CS/22928 - Intro to CV/blaaa/cv-final/data/external/test.csv"
    test_samples = pd.read_csv(input_csv)
    test_samples.rename(columns={'batch_id': 'scene_name', 'image_1_id': 'im1', 'image_2_id': 'im2'}, inplace=True)  # legacy rename for covisibilty compatibility
    pairs = test_samples # for original script compatibility

    if max_length > -1:
        pairs = pairs[0:np.min([len(pairs), max_length])]

    # Load the SuperPoint and SuperGlue models.
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print('Running on device \"{}\"'.format(device))

    
    matching = Matching(matching_config).eval().to(device)

    # Create the output directories if they do not exist already.
    input_dir = Path(input_dir)
    print('Looking for data in directory \"{}\"'.format(input_dir))

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print('Will write matches to directory \"{}\"'.format(output_dir))

    timer = AverageTimer(newline=True)
    
    for index, row in tqdm(pairs.iterrows(), desc="Going over dataset"):  #edited - change to iter on pd
        name0 = row['im1'] + '.jpg' # For name
        name1 = row['im2'] + '.jpg' # for namte
        scene_name = row['scene_name']
        scene_input_dir = input_dir / scene_name # path of images

        stem0, stem1 = Path(name0).stem, Path(name1).stem
        matches_path = output_dir / '{}-{}-matches.npz'.format(stem0, stem1) # changed to - from _ 
        eval_path = output_dir / '{}-{}-evaluation.npz'.format(stem0, stem1)

        if matches_path.exists():
            print("Skipping")
            continue

        do_match = True

        # If a rotation integer is provided (e.g. from EXIF data), use it:
        if len(row) >= 5:
            rot0, rot1 = int(row[2]), int(row[3])
        else:
            rot0, rot1 = 0, 0

        # Load the image pair.
        
        image0, inp0, scales0 = read_image(
            scene_input_dir / name0, device, resize, rot0, False)
        image1, inp1, scales1 = read_image(
            scene_input_dir / name1, device, resize, rot1, False)
        if image0 is None or image1 is None:
            print('Problem reading image pair: {} {}'.format(
                scene_input_dir/name0, scene_input_dir/name1))
            exit(1)
        timer.update('load_image')

        if do_match:
            # Perform the matching.
            pred = matching({'image0': inp0, 'image1': inp1})
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
            matches, conf = pred['matches0'], pred['matching_scores0']
            timer.update('matcher')

            # Write the matches to disk.
            out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,    ### Edited - write to disc
                           'matches': matches, 'match_confidence': conf,
                           'scene_name': scene_name, 'img0': scene_input_dir / name0, 'img1': scene_input_dir / name1}
            np.savez(str(matches_path), **out_matches)

        # Keep the matching keypoints.
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]


def predict_matrix_for_couple(npz_file, directory_path):
    data = np.load(os.path.join(directory_path, npz_file)) # load cached data
    exception = None

    # Extract matches, keypoints and general info
    matches = data['matches']
    kpts0 = data['keypoints0']
    kpts1 = data['keypoints1']
    match_conf = data['match_confidence']

    scene_name = data['scene_name']
    image_name0 = npz_file.split('-')[0]
    image_name1 = npz_file.split('-')[1]
    
    # Filter valid matches
    valid = []
    cnt = 0
    mkpts0 = []
    mkpts1 = []

    AMOUNT = 50
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
    
    try:
    # Estimate the fundamental matrix using the matched keypoints
        F, mask = cv.findFundamentalMat(
            mkpts0,
            mkpts1, 
            cv.USAC_MAGSAC, 
            ransacReprojThreshold=0.5, # was 0.5
            confidence=0.99999,
            maxIters=10000)
    except Exception as e:
        exception = type(e).__name__
        if len(mkpts0) <= 8:
            #print(f"Found {len(mkpts0)}")
            pass
        F = None

    # Append the results to the lists
    sample = f"{scene_name};{image_name0}-{image_name1}"
    
    if F is not None:
        if F.shape != (3,3):
            print(F.shape)
        F = F.reshape(-1)
        F = F[:9]
        
    return sample, F, exception


def create_output_file(cached_directory):

    # List all files in the cache directory
    files = os.listdir(cached_directory)

    # Filter out the npz files
    npz_files = [file for file in files if file.endswith('.npz')]
    samples = []
    fundamental_matrices = []
    exception_counts = {}


    # Iterate over all npz files
    for npz_file in tqdm(npz_files):
        sample, F, exp = predict_matrix_for_couple(npz_file, cached_directory)
        samples.append(sample)
        fundamental_matrices.append(F)
        if exp is None:
            continue
        if exp in exception_counts:
                exception_counts[exp] += 1
        else:
            exception_counts[exp] = 1
            #print(exp)
    total_errors = sum(exception_counts.values())
    print(F"total malformed {total_errors}")

    # Create a dataframe with the collected data
    df = pd.DataFrame({
        'sample_id': samples,
        'fundamental_matrix': fundamental_matrices
    })

    return df

def pretty_fundamental_matrix(row):
    if row is None:
        row = np.zeros(9) 
    # Convert array to a space-separated string in scientific notation
    print(type(row))
    return " ".join(f"{num:.5e}" for num in row)