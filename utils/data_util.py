from pathlib import Path
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass, field
import numpy as np
from utils import extractor_util as exu, preproc_utils as pu 

from tqdm import tqdm


@dataclass
class ImageData:
    name: str
    path: Path
    preproc_contents: any = None
    features: exu.ImageFeature = None
    for_exp: int = 1

@dataclass
class SceneData:
    images_dir: Path
    calibration: pd.DataFrame
    covisibility: pd.DataFrame
    image_data: Dict[str, ImageData] = field(default_factory=dict)

    #def __post_init__(self):
        #self.calibration['camera_intrinsics'] = self.calibration.camera_intrinsics.apply(lambda k: np.array([float(v) for v in k.split(' ')]).reshape([3, 3]))
       # self.calibration['rotation_matrix'] = self.calibration.rotation_matrix.apply(lambda k: np.array([float(v) for v in k.split(' ')]).reshape([3, 3]))
       # self.calibration['translation_vector'] = self.calibration.translation_vector.apply(lambda k: np.array([float(v) for v in k.split(' ')]))

@dataclass
class DatasetLoader:
    root_dir: str
    train_mode: bool = True
    scenes_data: Dict[str, SceneData] = field(default_factory=dict)
    test_samples: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self):
        self.root = Path(self.root_dir)
        self.train_dir = self.root / 'train'
        self.test_dir = self.root / 'test_images'
        if not self.train_mode:
            self.test_samples = self._load_test_samples()

    def _load_test_samples(self):
        test_samples_path = self.root / 'test.csv'
        
        test_samples = pd.read_csv(test_samples_path)
        test_samples.rename(columns={'batch_id': 'scene_name', 'image_1_id': 'im1', 'image_2_id': 'im2'}, inplace=True)
        
        return test_samples

    def _load_scene(self, scene_name: str) -> SceneData:
        scene_dir = self.train_dir / scene_name if self.train_mode else self.test_dir / scene_name
        images_dir = scene_dir / 'images' if self.train_mode else scene_dir

        # Check if calibration.csv exists
        calibration_file = scene_dir / 'calibration.csv'
        if calibration_file.exists():
            calibration = pd.read_csv(calibration_file, index_col=0).set_index('image_id')
            calibration['camera_intrinsics'] = calibration.camera_intrinsics.apply(
                lambda k: np.array([float(v) for v in k.split(' ')]).reshape([3, 3])
            )
            calibration['rotation_matrix'] = calibration.rotation_matrix.apply(
                lambda k: np.array([float(v) for v in k.split(' ')]).reshape([3, 3])
            )
            calibration['translation_vector'] = calibration.translation_vector.apply(
                lambda k: np.array([float(v) for v in k.split(' ')])
            )
        else:
            calibration = pd.DataFrame()  # Placeholder if calibration data is missing

        # Check if pair_covisibility.csv exists
        covisibility_file = scene_dir / 'pair_covisibility.csv'
        if covisibility_file.exists():
            covisibility = pd.read_csv(covisibility_file, index_col=0).set_index('pair')
        else:
            covisibility = pd.DataFrame(columns=['x'])  # Placeholder if covisibility data is missing

        scene_data = SceneData(
            images_dir=images_dir,
            calibration=calibration,
            covisibility=covisibility
        )

        self.scenes_data[scene_name] = scene_data
        return scene_data
    
    def _get_all_scenes(self) -> List[str]:
        if self.train_mode:
            return [d.name for d in self.train_dir.iterdir() if d.is_dir()]
        else:
            return [d.name for d in self.test_dir.iterdir() if d.is_dir()]

    def _load_all_scenes(self) -> Dict[str, SceneData]:
        scenes = self._get_all_scenes()
        for scene in scenes:
            scene_data = self._load_scene(scene)
            self.scenes_data[scene] = scene_data

        return self.scenes_data

    def load_all_dataset(self, preprocessor: pu.ImagePreprocessor, extractor: exu.FeatureExtractor, exclude_scenes: List = []):
        print("Loading image data and metadata")
        self.load_dataset_images(preprocessor, exclude_scenes)
        print("Extracting features from sences")
        self.extract_features(extractor)

    def load_dataset_images(self, preprocessor: pu.ImagePreprocessor, exclude_scenes: List = []):
        """
        Preprocesses the data in the dataset.
        first checks if the data has already been preprocessed, if not, preprocesses the data.

        :param dataset: The datasgit et to preprocess.
        :exclude_scenes: A list of scenes to exclude from preprocessing.

        """
        train_data = self._load_all_scenes()

        for scene in train_data:
            scene_data = train_data[scene]
            for img in scene_data.images_dir.iterdir():
                image_name = img.name.replace(img.suffix, '')
                if scene_data.image_data is None:
                        scene_data.image_data = {}

                preprocessed_img = preprocessor.process_image(img)
                scene_data.image_data[image_name] = ImageData(image_name, img, preprocessed_img)
    
    def extract_features(self, extractor: exu.FeatureExtractor):
        """
        Extracts features from the preprocessed images in the dataset.
        :param dataset: The dataset containing the preprocessed images and of course the scenes loaded.
        """
        for scene in tqdm(self.scenes_data, desc="Scenes"):
            scene_data_imgs = self.scenes_data[scene].image_data
            for img in tqdm(scene_data_imgs, desc="Images", leave=False):
                if scene_data_imgs[img].for_exp == 1:
                    scene_data_imgs[img].features = extractor.extract_features(scene_data_imgs[img].preproc_contents)

    
    def get_valid_pairs_to_match(self, max_pairs_per_scene, covisibility_threshold: float = 0.1):
        pass
