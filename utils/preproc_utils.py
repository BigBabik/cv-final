from dataclasses import dataclass
from typing import Tuple
import cv2 as cv
import numpy as np
from pathlib import Path
from typing import Union, List
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

@dataclass
class PreprocessConfig:
    size: Tuple[int, int] = (512, 512)
    normalize: bool = True
    greyscale: bool = False


@dataclass
class ImagePreprocessor:
    config: PreprocessConfig
    num_workers: int = 4
    
    def process_image(self, image_path: Union[str, Path]) -> np.ndarray:
        img = cv.imread(str(image_path))
        img = cv.resize(img, self.config.size)

        if self.config.normalize:
            img = img.astype(np.float32) / 255.0
        
        if self.config.greyscale:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            
        return img
    
    def process_batch(self, image_paths: List[Union[str, Path]], num_workers: int):
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            processed_images = list(executor.map(self.process_image, image_paths))
        return processed_images
