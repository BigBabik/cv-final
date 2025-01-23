#! /usr/bin/env python3
#
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import pandas as pd


from models.matching import Matching
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)

torch.set_grad_enabled(False)


def process_images(input_csv, input_dir, output_dir, do_viz=True, resize=[640, 480], max_length=-1, fast_viz=True):
    

    #with open(opt.input_pairs, 'r') as f:
        #pairs = [l.split() for l in f.readlines()]

    input_csv = "/share/project_data/test.csv"
    test_samples = pd.read_csv(input_csv)
    test_samples.rename(columns={'batch_id': 'scene_name', 'image_1_id': 'im1', 'image_2_id': 'im2'}, inplace=True)  # edited columns
    pairs = test_samples # #edited


    if max_length > -1:
        pairs = pairs[0:np.min([len(pairs), max_length])]

    # Load the SuperPoint and SuperGlue models.
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    print('Running inference on device \"{}\"'.format(device))

    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024
        },
        'superglue': {
            'weights': 'outdoor',
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
        }
    }
    matching = Matching(config).eval().to(device)

    # Create the output directories if they do not exist already.
    input_dir = Path(input_dir)
    print('Looking for data in directory \"{}\"'.format(input_dir))
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    print('Will write matches to directory \"{}\"'.format(output_dir))
    if do_viz:
        print('Will write visualization images to',
              'directory \"{}\"'.format(output_dir))

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
        viz_eval_path = output_dir / \
            '{}_{}_evaluation.{}'.format(stem0, stem1, "png")

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

        if do_viz:
            # Visualize the matches.
            color = cm.jet(mconf)
            text = [
                'SuperGlue',
                'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                'Matches: {}'.format(len(mkpts0)),
            ]
            if rot0 != 0 or rot1 != 0:
                text.append('Rotation: {}:{}'.format(rot0, rot1))

            # Display extra parameter info.
            k_thresh = matching.superpoint.config['keypoint_threshold']
            m_thresh = matching.superglue.config['match_threshold']
            small_text = [
                'Keypoint Threshold: {:.4f}'.format(k_thresh),
                'Match Threshold: {:.2f}'.format(m_thresh),
                'Image Pair: {}:{}'.format(stem0, stem1),
            ]

            make_matching_plot(
                image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
                text, viz_path, show_keypoints=True,
                       fast_viz=fast_viz, opencv_display=False,
                       opencv_title='matches', small_text=[])

            timer.update('viz_match')

      