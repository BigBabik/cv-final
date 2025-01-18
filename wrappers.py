
from utils import data_util as du, preproc_utils as pu, extractor_util as exu, estimator_util as esu
from utils.data_util import ImageData, SceneData, DatasetLoader
from utils.preproc_utils import ImagePreprocessor
from utils.extractor_util import FeatureExtractor
from utils.matcher_utils import FeatureMatcher
from typing import List
import json
import pandas as pd
import numpy as np
import random
#from utils import general_utils as gu
from utils import evaluation_util as evu
from tqdm import tqdm
import time
from utils.general_utils import normalize_keypoints, pack_coords, unpack_coords
 
SUBMISSION_COLS = ['sample_id', 'fundamental_matrix', 'mask', 'inliers1', 'inliers2']



def sample_pairs_for_run(dataset: DatasetLoader, max_pairs_per_scene: int, covisibility_threshold: float = 0.1):
    """
    Called after preprocessing so dataset already has scenes_data populated
    """
    if not dataset.train_mode:
        dataset.test_samples.loc[:, 'for_exp'] = np.nan

    for scene in dataset.scenes_data:
        scene_data = dataset.scenes_data[scene]

        if dataset.train_mode:
            valid_pairs_df = scene_data.covisibility[scene_data.covisibility['covisibility'] > covisibility_threshold]
        else:
            valid_pairs_df = dataset.test_samples
            # Filter test_samples for current scene using scene_name column
            valid_pairs_df = dataset.test_samples[dataset.test_samples['scene_name'] == scene]

        valid_pairs_indices = valid_pairs_df.index.tolist()
        
        if dataset.train_mode:
            scene_data.covisibility.loc[:, 'for_exp'] = np.nan

        print(f'[+] Processing scene "{scene}": found {len(valid_pairs_indices)} pairs (will keep {min(len(valid_pairs_indices), max_pairs_per_scene)})', flush=True)

        random.shuffle(valid_pairs_indices)
        valid_pairs_indices = valid_pairs_indices[:max_pairs_per_scene]

        if dataset.train_mode:
            scene_data.covisibility.loc[valid_pairs_indices, 'for_exp'] = 1
            sampled_df = scene_data.covisibility.loc[valid_pairs_indices]
        else:
            dataset.test_samples.loc[valid_pairs_indices, 'for_exp'] = 1
            sampled_df = dataset.test_samples.loc[valid_pairs_indices]
        
        valid_imgs = set(sampled_df[['im1', 'im2']].values.flatten())

        
        for img in scene_data.image_data:
            if img not in valid_imgs:
                scene_data.image_data[img].for_exp = 0



def match_features(dataset: DatasetLoader, matcher: FeatureMatcher, covisibility_threshold: float = 0.1):
    """
    Matches features between images in the dataset with optimized performance.
    :param dataset: The dataset containing the images with extracted features.
    """
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    
    def process_image_pair(args):
        index, row, scene_data, train_mode = args
        
        # Cache image data access
        img1 = scene_data.image_data[row['im1']]
        img2 = scene_data.image_data[row['im2']]
        
        # Batch process matches
        matches = matcher.match_features(img1.features, img2.features)
        valid, kp1, kp2 = matcher.filter_lowe_matches(matches, img1.features, img2.features)
        
        # Vectorize keypoint normalization
        if not scene_data.calibration.empty:
            k1_pts = np.array([p.pt for p in kp1])
            k2_pts = np.array([p.pt for p in kp2])
            k1n = normalize_keypoints(k1_pts, scene_data.calibration.loc[img1.name].camera_intrinsics)
            k2n = normalize_keypoints(k2_pts, scene_data.calibration.loc[img2.name].camera_intrinsics)
        else:
            k1n = [p.pt for p in kp1]
            k2n = [p.pt for p in kp2]
        
        result = {
            'index': index,
            'kp1': pack_coords(k1n),
            'kp2': pack_coords(k2n)
        }
        
        if not train_mode:
            result.update({
                'im1': img1.name,
                'im2': img2.name
            })
            
        return result

    start_time = time.time()
    
    for scene in dataset.scenes_data:
        scene_data = dataset.scenes_data[scene]
        print(f"Matching features for scene: {scene}")
        
        # Filtering logic moved outside the loop
        if dataset.train_mode:
            valid_pairs = scene_data.covisibility[scene_data.covisibility['covisibility'] > covisibility_threshold]
            if 'for_exp' in scene_data.covisibility.columns:
                valid_pairs = valid_pairs.dropna(subset=['for_exp'])
        else:
            valid_pairs = dataset.test_samples[dataset.test_samples['scene_name'] == scene]
            if 'for_exp' in dataset.test_samples.columns:
                valid_pairs = valid_pairs.dropna(subset=['for_exp'])
        
        print(f"In matcher there are {len(valid_pairs)} valid pairs to estimate for")
        
        # Prepare batch processing arguments
        process_args = [(index, row, scene_data, dataset.train_mode) 
                       for index, row in valid_pairs.iterrows()]
        
        # Process image pairs in parallel
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(process_image_pair, process_args))
        
        # Batch update the dataframe
        for result in results:
            index = result['index']
            scene_data.covisibility.loc[index, 'keypoints1'] = result['kp1']
            scene_data.covisibility.loc[index, 'keypoints2'] = result['kp2']
            
            if not dataset.train_mode:
                scene_data.covisibility.loc[index, 'im1'] = result['im1']
                scene_data.covisibility.loc[index, 'im2'] = result['im2']
        
        end_time = time.time()
        print(f"Total time taken for matching features in scene {scene}: {end_time - start_time:.2f} seconds")
        start_time = time.time()


def estimate_fundamental_matrix(dataset: DatasetLoader, estimator: esu.FundamentalMatrixEstimator):
    """
    Estimates the fundamental matrix between images in the dataset.
    :param dataset: The dataset containing the images with matched features.
    """
    # make sure to only estimate for valid images and how to construct the submision file from this
    # maybe mark valid in covisibility for this run? or jus tfilter again and take the uniqwue id from there
    submissions_list = []

    for scene in dataset.scenes_data:
        scene_data = dataset.scenes_data[scene]
        print(f"Estimating fundamental matrix for pairs in scene: {scene}")

        valid_pairs = scene_data.covisibility.dropna(subset=['keypoints1', 'keypoints2'])
        if 'for_exp' in scene_data.covisibility.columns:
            valid_pairs = valid_pairs.dropna(subset=['for_exp'])

        for index, row in valid_pairs.iterrows():
            img1 = scene_data.image_data[row['im1']]
            img2 = scene_data.image_data[row['im2']]
            
            # Get the keypoints from the string representation
            kp1 = np.array(unpack_coords(row['keypoints1']))
            kp2 = np.array(unpack_coords(row['keypoints2']))

            # Estimate the fundamental matrix
            estimator.keypoints1 = kp1
            estimator.keypoints2 = kp2
            try:
                estimated_fund, mask = estimator.estimate()
            except Exception as e:
                print(e.__traceback__)
                continue

            inliers1 = estimator.keypoints1[estimator.mask.ravel() == 1]
            inliers2 = estimator.keypoints2[estimator.mask.ravel() == 1]

            sample_id = f"{scene};{row['im1']}-{row['im2']}"

            submissions_list.append([sample_id, estimated_fund, mask, inliers1, inliers2])

    return pd.DataFrame(submissions_list, columns=SUBMISSION_COLS)


def get_camera_intrinsics_for_pair(sample_id, scene_data):
    img1, img2 = sample_id.split(';')[-1].split('-')
    K1 = scene_data.calibration.loc[img1].camera_intrinsics
    K2 = scene_data.calibration.loc[img2].camera_intrinsics
    return K1, K2

def get_keypoints_for_pair(sample_id, scene_data):
    i1, i2 = sample_id.split(';')[-1].split('-')
    kp1, kp2 = scene_data.covisibility.loc['-'.join([i1, i2])][['keypoints1', 'keypoints2']]
    kp1 = np.array(unpack_coords(kp1))
    kp2 = np.array(unpack_coords(kp2))
    return kp1, kp2

def get_gt_rt_for_pair(sample_id, scene_data):
    img1, img2 = sample_id.split(';')[-1].split('-')
    cols = ['rotation_matrix', 'translation_vector']

    R1_gt, T1_gt = scene_data.calibration.loc[img1][cols]
    R2_gt, T2_gt = scene_data.calibration.loc[img2][cols]

    return R1_gt, T1_gt.reshape((3, 1)), R2_gt, T2_gt.reshape((3, 1))

def evaluate_results(dataset: DatasetLoader, results: pd.DataFrame, scaling: pd.DataFrame, thresholds_q: np.linspace, thresholds_t: np.geomspace) -> pd.DataFrame:
    rel_list = []
    for scene in dataset.scenes_data: #find a better way to do this because I apply for all of the DF even with rows that aren't related
        print(f"Evaluating for {scene}")
        scene_data = dataset.scenes_data[scene]
        scale = scaling.loc[scene].scaling_factor

        rel_results = results[results['sample_id'].str.startswith(scene)]
        if len(rel_results) == 0:
            continue
        rel_results = rel_results.dropna(subset=['fundamental_matrix'])

        rel_results[['K1', 'K2']] = rel_results['sample_id'].apply(lambda x: pd.Series(get_camera_intrinsics_for_pair(x, scene_data)))
        rel_results[['kp1', 'kp2']] = rel_results['sample_id'].apply(lambda x: pd.Series(get_keypoints_for_pair(x, scene_data)))

        rel_results[['E', 'R', 'T']] = rel_results.apply(lambda row: pd.Series(evu.compute_essential_matrix(np.array(row['fundamental_matrix']),
                                                np.array(row['K1']), np.array(row['K2']), row['kp1'], row['kp2'])), axis=1)
        
        rel_results['q'] = rel_results['R'].apply(lambda r: evu.quaternion_from_matrix(r))

        rel_results[['R1_gt', 'T1_gt', 'R2_gt', 'T2_gt']] = rel_results['sample_id'].apply(lambda x: pd.Series(get_gt_rt_for_pair(x, scene_data)))
        rel_results['dR_gt'] = rel_results.apply(lambda row: np.dot(row['R2_gt'], row['R1_gt'].T), axis=1)
        rel_results['dT_gt'] = rel_results.apply(lambda row: (row['T2_gt'] - np.dot(row['dR_gt'], row['T1_gt'])).flatten(), axis=1)
        rel_results['q_gt'] = rel_results['dR_gt'].apply(lambda dR_gt: evu.quaternion_from_matrix(dR_gt))
        rel_results['q_gt'] = rel_results['q_gt'].apply(lambda q_gt: q_gt / (np.linalg.norm(q_gt) + np.finfo(float).eps))
        
        rel_results[['err_q', 'err_t']] = rel_results.apply(lambda row: 
            pd.Series(evu.compute_error_for_pair(row['q_gt'], row['dT_gt'], row['q'], row['T'], scale)), 
            axis=1)

        maa, acc, acc_q, acc_t = evu.compute_mean_average_acc(rel_results.err_q, rel_results.err_t, thresholds_q, thresholds_t)

        rel_results.loc[:, 'maa'] = maa
        rel_results.loc[:, 'acc'] = [acc.tolist()] * len(rel_results)
        rel_results.loc[:, 'acc_q'] = [acc_q.tolist()] * len(rel_results)
        rel_results.loc[:, 'acc_t'] = [acc_t.tolist()] * len(rel_results)

        rel_list.append(rel_results)

    all_rel_results = pd.concat(rel_list, axis=0)
    results = pd.concat([results, all_rel_results[['E', 'R', 'T', 'q', 'R1_gt', 'T1_gt', 'R2_gt', 'T2_gt', 'dR_gt', 'dT_gt', 'q_gt', 'err_q', 'err_t', 'maa', 'acc', 'acc_q', 'acc_t']]], axis=1)    

    return results