
from utils import data_util as du, preproc_utils as pu, extractor_util as exu, estimator_util as esu
from utils.data_util import ImageData, SceneData, DatasetLoader
from utils.preproc_utils import ImagePreprocessor, PreprocessConfig
from utils.extractor_util import FeatureExtractor, MatcherConfig, FeatureMatcher
from typing import List
import json
import pandas as pd
import numpy as np
import random
#from utils import general_utils as gu
from utils import evaluation_util as evu
 
SUBMISSION_COLS = ['sample_id', 'fundamental_matrix', 'mask', 'inliers1', 'inliers2']

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

def parse_config(config_file):
    """
    Parses the configuration file and returns the configuration objects for each stage.
    :param config_file: The configuration file to parse.

    :return: The configuration objects.
    """
    with open(config_file, 'r') as file:
        config = json.load(file)
    
    preprocessor_config = PreprocessConfig(**config['preprocessor'])
    return preprocessor_config


def load_dataset(dataset_dir):
    """
    Loads the dataset from the specified directory.
    :param dataset_dir: The directory containing the dataset (will be the external data directory).

    :return: A DatasetLoader object containing the dataset.
    """
    return DatasetLoader(root_dir=dataset_dir)

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
        print(valid_pairs_indices)

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
    

def preprocess_data(dataset: DatasetLoader, preprocessor: ImagePreprocessor, exclude_scenes: List = []):
    """
    Preprocesses the data in the dataset.
    first checks if the data has already been preprocessed, if not, preprocesses the data.

    :param dataset: The datasgit et to preprocess.
    :exclude_scenes: A list of scenes to exclude from preprocessing.

    """
    train_data = dataset.load_all_scenes()

    for scene in train_data:
        scene_data = train_data[scene]
        for img in scene_data.images_dir.iterdir():
            image_name = img.name.replace(img.suffix, '')
            if scene_data.image_data is None:
                    scene_data.image_data = {}

            preprocessed_img = preprocessor.process_image(img)
            scene_data.image_data[image_name] = du.ImageData(image_name, img, preprocessed_img)

    return dataset

def extract_features(dataset: DatasetLoader, extractor: FeatureExtractor):
    """
    Extracts features from the preprocessed images in the dataset.
    :param dataset: The dataset containing the preprocessed images and of course the scenes loaded.
    """
    for scene in dataset.scenes_data:
        scene_data_imgs = dataset.scenes_data[scene].image_data
        for img in scene_data_imgs: 
            if scene_data_imgs[img].for_exp == 1:
                scene_data_imgs[img].features = extractor.extract_features(scene_data_imgs[img].preproc_contents)

    return dataset


def match_features(dataset: DatasetLoader, matcher: FeatureMatcher, covisibility_threshold: float = 0.1):
    """
    Matches features between images in the dataset.
    :param dataset: The dataset containing the images with extracted features.
    """
    
    for scene in dataset.scenes_data:
        scene_data = dataset.scenes_data[scene]

        print(f"Matching features for scene: {scene}")
        
        # Now filter
        if dataset.train_mode:
            valid_pairs = scene_data.covisibility[scene_data.covisibility['covisibility'] > covisibility_threshold]
            if 'for_exp' in scene_data.covisibility.columns:
                valid_pairs = valid_pairs.dropna(subset=['for_exp'])
        else:
            valid_pairs = dataset.test_samples[dataset.test_samples['scene_name'] == scene]
            if 'for_exp' in dataset.test_samples.columns:
                valid_pairs = valid_pairs.dropna(subset=['for_exp'])

        print(f"In matcher there are {len(valid_pairs)} valid pairs to estimate for")
        
        for index, row in valid_pairs.iterrows():
            img1 = scene_data.image_data[row['im1']]  # Note: row[1] to access the Series
            img2 = scene_data.image_data[row['im2']]

            print(img1.name, img2.name)
            matches = matcher.match_features(img1.features, img2.features)
            valid, kp1, kp2 = matcher.filter_lowe_matches(matches, img1.features, img2.features)

            if not scene_data.calibration.empty:
                k1n = normalize_keypoints([p.pt for p in kp1], scene_data.calibration.loc[img1.name].camera_intrinsics)
                k2n = normalize_keypoints([p.pt for p in kp2], scene_data.calibration.loc[img2.name].camera_intrinsics)
            else:
                k1n = [p.pt for p in kp1]
                k2n = [p.pt for p in kp2]
            
            kp1_update = pack_coords(k1n)
            kp2_update = pack_coords(k2n)
            
            # Update one row at a time - IMPLEMENT BATCH UPDATE AND SAY IF UPDATING A NULL LIST - FAILED TO MATCH
            scene_data.covisibility.loc[index, 'keypoints1'] = kp1_update
            scene_data.covisibility.loc[index, 'keypoints2'] = kp2_update


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
            estimated_fund, mask = estimator.estimate()

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