from pathlib import Path
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass
import os

@dataclass
class SceneData:
    images_dir: Path
    # image_files: List[str] = os.listdir(images_dir)
    calibration: pd.DataFrame
    covisibility: pd.DataFrame

class DatasetLoader:
    def __init__(self, root_dir: str):
        self.root = Path(root_dir)
        self.train_dir = self.root / 'train'
        self.test_dir = self.root / 'external/test_images'
    
    def load_scene(self, scene_name: str) -> SceneData:
        scene_dir = self.train_dir / scene_name
        return SceneData(
            images_dir=scene_dir / 'images',
            calibration=pd.read_csv(scene_dir / 'calibration.csv', index_col=0),
            covisibility=pd.read_csv(scene_dir / 'pair_covisibility.csv', index_col=0)
        )
    
    def get_all_scenes(self) -> List[str]:
        return [d.name for d in self.train_dir.iterdir() if d.is_dir()]
    
    def load_all_scenes(self) -> Dict[str, SceneData]:
        return {scene: self.load_scene(scene) 
                for scene in self.get_all_scenes()}
