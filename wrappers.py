import os
from utils import data_util as du, preproc_utils as pu, extractor_util as exu, estimator_util as esu
from utils.data_util import ImageData, SceneData, DatasetLoader
from utils.preproc_utils import ImagePreprocessor, PreprocessConfig
from utils.extractor_util import FeatureExtractor, MatcherConfig, FeatureMatcher
from typing import List
import json
import pandas as pd
import numpy as np


SUBMISSION_COLS = ['sample_id', 'fundamental_matrix']


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

def preprocess_data(dataset: DatasetLoader, preprocessor: ImagePreprocessor, exclude_scenes: List = []):
    """
    Preprocesses the data in the dataset.
    first checks if the data has already been preprocessed, if not, preprocesses the data.

    :param dataset: The dataset to preprocess.
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
        scene_data = dataset.scenes_data[scene]
        for img in scene_data.image_data: 
            scene_data.image_data[img].features = extractor.extract_features(scene_data.image_data[img].preproc_contents)  

    return dataset

def pack_coords(coord_list: List):
    """
    :param coord_list: listof KeyPoint objs
    """
    return ';'.join([str(p.pt) for p in coord_list])

def unpack_coords(string_of_coords: str):
    points = string_of_coords.split(';')
    return ([eval(p) for p in points])


def match_features(dataset: DatasetLoader, matcher: FeatureMatcher, covisibility_threshold: float = 0.1):
    """
    Matches features between images in the dataset.
    :param dataset: The dataset containing the images with extracted features.
    """
    
    for scene in dataset.scenes_data:
        scene_data = dataset.scenes_data[scene]
        print(f"Matching features for scene: {scene}")
        
        # Now filter
        valid_pairs = scene_data.covisibility[scene_data.covisibility['covisibility'] > covisibility_threshold]
        
        for index, row in valid_pairs.iterrows():
            img1 = scene_data.image_data[row['im1']]  # Note: row[1] to access the Series
            img2 = scene_data.image_data[row['im2']]

            matches = matcher.match_features(img1.features, img2.features)
            valid, kp1, kp2 = matcher.filter_lowe_matches(matches, img1.features, img2.features)

            kp1_update = pack_coords(kp1)
            kp2_update = pack_coords(kp2)
            #print(kp1_update, kp2_update)
            
            # Update one row at a time - IMPLEMENT BATCH UPDATE AND SAY IF UPDATING A NULL LIST - FAILED TO MATCH
            scene_data.covisibility.loc[index, 'keypoints1'] = kp1_update
            scene_data.covisibility.loc[index, 'keypoints2'] = kp2_update
        break


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

        for index, row in valid_pairs.iterrows():
            img1 = scene_data.image_data[row['im1']]
            img2 = scene_data.image_data[row['im2']]
            
            # Get the keypoints from the string representation
            kp1 = np.array(unpack_coords(row['keypoints1']))
            kp2 = np.array(unpack_coords(row['keypoints2']))

            
            # Estimate the fundamental matrix
            estimator.keypoints1 = kp1
            estimator.keypoints2 = kp2
            estimated_fund = estimator.estimate()

            sample_id = f"{scene};{row['pair']}"

            submissions_list.append([sample_id, estimated_fund])

        break

    return pd.DataFrame(submissions_list, columns=SUBMISSION_COLS)

