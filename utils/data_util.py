from pathlib import Path
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass, field
import numpy as np
from utils import extractor_util as exu

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
    scenes_data: Dict[str, SceneData] = field(default_factory=dict)
    train_mode: bool = True

    def __post_init__(self):
        self.root = Path(self.root_dir)
        self.train_dir = self.root / 'train'
        self.test_dir = self.root / 'test_images'
    
    def load_scene(self, scene_name: str) -> SceneData:
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
            covisibility = pd.DataFrame()  # Placeholder if covisibility data is missing

        scene_data = SceneData(
            images_dir=images_dir,
            calibration=calibration,
            covisibility=covisibility
        )

        self.scenes_data[scene_name] = scene_data
        return scene_data
    
    # -----

        if self.train_mode:
            scene_dir = self.train_dir / scene_name
            scene_data = SceneData(
            images_dir=scene_dir / 'images',
            calibration=pd.read_csv(scene_dir / 'calibration.csv', index_col=0).set_index('image_id'), # will crash
            covisibility=pd.read_csv(scene_dir / 'pair_covisibility.csv', index_col=0).set_index('pair'), # will crash
        )
        else:
            scene_dir = self.test_dir / scene_name
            scene_data = SceneData(
            images_dir=scene_dir / 'images',
            calibration=pd.DataFrame(),
            covisibility=pd.DataFrame()
            )


        self.scenes_data[scene_name] = scene_data
        return scene_data
    
    def get_all_scenes(self) -> List[str]:
        if self.train_mode:
            return [d.name for d in self.train_dir.iterdir() if d.is_dir()]
        else:
            return [d.name for d in self.test_dir.iterdir() if d.is_dir()]

    def load_all_scenes(self) -> Dict[str, SceneData]:
        self.scenes_data = {scene: self.load_scene(scene) 
                            for scene in self.get_all_scenes()}
        return self.scenes_data
