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
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib
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
    
    for index, row in pairs.iterrows():  #edited - change to iter on pd
        name0 = row['im1'] + '.jpg' # For name
        name1 = row['im2'] + '.jpg' # for namte
        scene_name = row['scene_name']
        scene_input_dir = input_dir / scene_name # path of images

        stem0, stem1 = Path(name0).stem, Path(name1).stem
        matches_path = output_dir / '{}-{}-matches.npz'.format(stem0, stem1) # changed to - from _ 
        eval_path = output_dir / '{}-{}-evaluation.npz'.format(stem0, stem1)
        viz_path = output_dir / '{}-{}-matches.{}'.format(stem0, stem1, "png")

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