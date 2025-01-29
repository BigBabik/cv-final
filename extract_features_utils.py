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
import cv2
import os

from models.matching import Matching
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)

torch.set_grad_enabled(False)


def create_super_matching(device):
    config = {
        'superpoint': {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
        },
        'superglue': {
        'descriptor_dim': 256,
        'weights': 'outdoor',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
        'max_keypoints': -1 
        }
    }
    matching = Matching(config).eval().to(device)

    return matching

def process_superglue(
        images_directory,
        image0_name,
        image1_name,
        device,
        matching,
        scene_name,
        output_path=None):
    # Load the image pair.
    image0_path = f"{images_directory}/{image0_name}.jpg"
    image1_path = f"{images_directory}/{image1_name}.jpg"
    no_rotation = 0
    no_resize = (0,0)

    image0, inp0, scales0 = read_image(
        image0_path, device, no_resize, no_rotation, False)
    image1, inp1, scales1 = read_image(
        image1_path, device, no_resize, no_rotation, False)
    if image0 is None or image1 is None:
        print('Problem reading image pair: {} {}'.format(
            image0_path, image1_path))
        raise Exception("Missing files")

    pred = matching({'image0': inp0, 'image1': inp1})
    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']
    
    if output_path:
        out_file_path = f"{output_path}/{image0_name}-{image1_name}-matches.npz"
    # Write the matches to disk.
        out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,    ### Edited - write to disc
                        'matches': matches, 'match_confidence': conf,
                        'scene_name': scene_name, 'img0': image1_path, 'img1': image1_path}
        np.savez(str(out_file_path), **out_matches)

    # Keep the matching keypoints.
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    mconf = conf[valid]

    return kpts0, kpts1, matches, conf

def process_images(
        input_csv,
        input_dir,
        output_dir,
        do_viz=True,
        resize=[-1, -1],
        max_length=-1,
         fast_viz=True):

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

    best_config = {
        'superpoint': {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
        },
        'superglue': {
        'descriptor_dim': 256,
        'weights': 'outdoor',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
        'max_keypoints': -1 
        }
    }
    new_config = {
        'superpoint': {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.05, # was 0.005
        'max_keypoints': -1,
        'remove_borders': 4,
        },
        'superglue': {
        'descriptor_dim': 256,
        'weights': 'outdoor',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.5,
        'max_keypoints': -1 
        }
    }

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
    matching = Matching(new_config).eval().to(device)

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

      

############ For sift

def load_image(image_path,
               use_color_image=False,
               input_width=512,
               crop_center=True,
               force_rgb=False):
    '''
    Loads image and do preprocessing.

    Parameters
    ----------
    image_path: Fullpath to the image.
    use_color_image: Flag to read color/gray image
    input_width: Width of the image for scaling
    crop_center: Flag to crop while scaling
    force_rgb: Flag to convert color image from BGR to RGB

    Returns
    -------
    Tuple of (Color/Gray image, scale_factor)
    '''

    # Assuming all images in the directory are color images
    image = cv2.imread(image_path)
    if not use_color_image:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif force_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Crop center and resize image into something reasonable
    scale_factor = 1.0
    if crop_center:
        rows, cols = image.shape[:2]
        if rows > cols:
            cut = (rows - cols) // 2
            img_cropped = image[cut:cut + cols, :]
        else:
            cut = (cols - rows) // 2
            img_cropped = image[:, cut:cut + rows]
        scale_factor = float(input_width) / float(img_cropped.shape[0])
        image = cv2.resize(img_cropped, (input_width, input_width))

    return (image, scale_factor)

def convert_opencv_kp_desc(kp, desc, num_kp):
    '''Converts opencv keypoints and descriptors to benchmark format.

    Parameters
    ----------
    kp: list
        List of keypoints in opencv format
    desc: list
        List of descriptors in opencv format
    num_kp: int
        Number of keypoints to extract per image
    '''

    # Convert OpenCV keypoints to list data structure used for the benchmark.
    kp = opencv_kp_list_2_kp_list(kp)

    # Sort keypoints and descriptors by keypoints response
    kp_desc = [(_kp, _desc)
               for _kp, _desc in sorted(zip(kp, desc),
                                        key=lambda x: x[0][IDX_RESPONSE])]
    kp_sorted = [kp for kp, desc in kp_desc]
    desc_sorted = [desc for kp, desc in kp_desc]
    # Reverse for descending order
    keypoints = kp_sorted[::-1]
    descriptors = desc_sorted[::-1]
    # Remove redundant points
    cur_num_kp = len(keypoints)
    keypoints = keypoints[:min(cur_num_kp, num_kp)]
    descriptors = descriptors[:min(cur_num_kp, num_kp)]

    return keypoints, descriptors

def run(img_path, cfg, kp_name, desc_name, num_kp = 1024):
    '''Wrapper over OpenCV SIFT.

    Parameters
    ----------
    img_path (str): Path to images. 
    cfg: (Namespace): Configuration.

    Valid keypoint methods: "sift-def" (standard detection threshold)
    and "sift-lowth" (lowered detection threshold to extract 8000 features).
    Optional suffixes: "-clahe" (applies CLAHE over the image).

    Valid descriptors methods: "sift" and "rootsift".
    Optional suffixes: "-clahe" (applies CLAHE over the image), "upright"
    (sets keypoint orientations to 0, removing duplicates).
    '''



    if kp_name == 'sift-def':
        use_lower_det_th = False
        use_clahe_det = False
    elif kp_name == 'sift-lowth':
        use_lower_det_th = True
        use_clahe_det = False
    elif kp_name == 'sift-def-clahe':
        use_lower_det_th = False
        use_clahe_det = True
    elif kp_name == 'sift-lowth-clahe':
        use_lower_det_th = True
        use_clahe_det = True
    else:
        raise ValueError('Unknown detector')

    if desc_name == 'sift':
        use_rootsift = False
        use_clahe_desc = False
        use_upright = False
        use_upright_minus_minus = False
    elif desc_name == 'rootsift':
        use_rootsift = True
        use_clahe_desc = False
        use_upright = False
        use_upright_minus_minus = False
    elif desc_name == 'sift-clahe':
        use_rootsift = False
        use_clahe_desc = True
        use_upright = False
        use_upright_minus_minus = False
    elif desc_name == 'rootsift-clahe':
        use_rootsift = True
        use_clahe_desc = True
        use_upright = False
        use_upright_minus_minus = False
    elif desc_name == 'sift-upright':
        use_rootsift = False
        use_clahe_desc = False
        use_upright = True
        use_upright_minus_minus = False
    elif desc_name == 'sift-upright--':
        use_rootsift = False
        use_clahe_desc = False
        use_upright = True
        use_upright_minus_minus = True
    elif desc_name == 'rootsift-upright':
        use_rootsift = True
        use_clahe_desc = False
        use_upright = True
        use_upright_minus_minus = False
    elif desc_name == 'rootsift-upright--':
        use_rootsift = True
        use_clahe_desc = False
        use_upright = True
        use_upright_minus_minus = True
    elif desc_name == 'sift-clahe-upright':
        use_rootsift = False
        use_clahe_desc = True
        use_upright = True
        use_upright_minus_minus = False
    elif desc_name == 'sift-clahe-upright--':
        use_rootsift = False
        use_clahe_desc = True
        use_upright = True
        use_upright_minus_minus = True
    elif desc_name == 'rootsift-clahe-upright':
        use_rootsift = True
        use_clahe_desc = True
        use_upright = True
        use_upright_minus_minus = False
    elif desc_name == 'rootsift-clahe-upright--':
        use_rootsift = True
        use_clahe_desc = True
        use_upright = True
        use_upright_minus_minus = True
    else:
        raise ValueError('Unknown descriptor')

    # print('Extracting SIFT features with'
    #         ' use_lower_det_th={},'.format(use_lower_det_th),
    #         ' use_clahe_det={},'.format(use_clahe_det),
    #         ' use_rootsift={},'.format(use_rootsift),
    #         ' use_clahe_desc={},'.format(use_clahe_desc),
    #         ' use_upright={}'.format(use_upright))

    # Initialize feature extractor
    NUM_FIRST_DETECT = 100000000
    if use_upright_minus_minus:
        NUM_FIRST_DETECT = num_kp
    if use_lower_det_th:
        feature = cv2.xfeatures2d.SIFT_create(NUM_FIRST_DETECT, 
                                              contrastThreshold=-10000,
                                              edgeThreshold=-10000)
    else:
        feature = cv2.xfeatures2d.SIFT_create(NUM_FIRST_DETECT)


    img_det, _ = load_image(img_path,
                            use_color_image=False,
                            crop_center=False)

    img_desc, _ = load_image(img_path,
                                 use_color_image=False,
                                 crop_center=False)

    # Get keypoints
    kp = feature.detect(img_det, None)

    # Compute descriptors
    if use_upright:
        unique_kp = []
        for i, x in enumerate(kp):
            if i > 0:
                if x.response == kp[i - 1].response:
                    continue
            x.angle = 0
            unique_kp.append(x)
        unique_kp, unique_desc = feature.compute(img_desc, unique_kp, None)
        top_resps = np.array([x.response for x in unique_kp])
        idxs = np.argsort(top_resps)[::-1]
        kp = np.array(unique_kp)[idxs[:min(len(unique_kp), num_kp)]]
        desc = unique_desc[idxs[:min(len(unique_kp), num_kp)]]
    else:
        kp, desc = feature.compute(img_desc, kp, None)

    # Use root-SIFT
    if use_rootsift:
        desc /= desc.sum(axis=1, keepdims=True) + 1e-8
        desc = np.sqrt(desc)

    # Convert opencv keypoints into our format
    kp, desc = convert_opencv_kp_desc(kp, desc, num_kp)
    keypoints  = [p[0:2] for p in kp]
    #result['scale'] = [p[2] for p in kp]
    #result['angle'] = [p[3] for p in kp]
    scores = [p[4] for p in kp]
    
    return keypoints, desc, scores