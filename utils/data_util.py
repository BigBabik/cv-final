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

    def __post_init__(self):
        self.calibration['camera_intrinsics'] = self.calibration.camera_intrinsics.apply(lambda k: np.array([float(v) for v in k.split(' ')]).reshape([3, 3]))
        self.calibration['rotation_matrix'] = self.calibration.rotation_matrix.apply(lambda k: np.array([float(v) for v in k.split(' ')]).reshape([3, 3]))
        self.calibration['translation_vector'] = self.calibration.translation_vector.apply(lambda k: np.array([float(v) for v in k.split(' ')]))

@dataclass
class DatasetLoader:
    root_dir: str
    scenes_data: Dict[str, SceneData] = field(default_factory=dict)

    def __post_init__(self):
        self.root = Path(self.root_dir)
        self.train_dir = self.root / 'train'
        self.test_dir = self.root / 'external/test_images'
    
    def load_scene(self, scene_name: str) -> SceneData:
        scene_dir = self.train_dir / scene_name

        scene_data = SceneData(
            images_dir=scene_dir / 'images',
            calibration=pd.read_csv(scene_dir / 'calibration.csv', index_col=0).set_index('image_id'),
            covisibility=pd.read_csv(scene_dir / 'pair_covisibility.csv', index_col=0).set_index('pair'),
        )

        self.scenes_data[scene_name] = scene_data
        return scene_data
    
    def get_all_scenes(self) -> List[str]:
        return [d.name for d in self.train_dir.iterdir() if d.is_dir()]

    def load_all_scenes(self) -> Dict[str, SceneData]:
        self.scenes_data = {scene: self.load_scene(scene) 
                            for scene in self.get_all_scenes()}
        return self.scenes_data
