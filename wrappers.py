import os
from utils import data_util as du, preproc_utils as pu, extractor_util as exu, estimator_util as esu
from utils.data_util import ImageData, SceneData, DatasetLoader
from utils.preproc_utils import ImagePreprocessor, PreprocessConfig
from typing import List
import json



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
    return DatasetLoader(dataset_dir)

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

    return train_data