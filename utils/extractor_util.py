from dataclasses import dataclass
from typing import Tuple, Optional
import cv2 as cv
import numpy as np
from enum import Enum

class ExtrectorType(Enum):
    SIFT = 1
    ORB = 2
    CUSTOM = 3

def _create_default_sift(
            nfeatures: int = 0,
            contrastThreshold: float = 0.04,
            edgeThreshold: float = 10.0,
            sigma: float = 1.6):
    
    sift_extractor = cv.SIFT_create(
                nfeatures=nfeatures,
                contrastThreshold=contrastThreshold,
                edgeThreshold=edgeThreshold,
                sigma=sigma)
    
    return sift_extractor

@dataclass
class ImageFeature:
    keypoints: list
    descriptors: Optional[Tuple]

class FeatureExtractor:
    def __init__(self, extrector_type :ExtrectorType):
        match extrector_type:
            case ExtrectorType.SIFT:
                self.extractor = _create_default_sift();
            case default:
                raise ValueError("No feature extractor specified in config.")
            

    def extract_features(self, image: np.ndarray) -> ImageFeature:    
        """
        Extracts features from an image using the algorithm provided to the extractor.

        :param image: The image to extract features from, as a numpy array from the preprocessing step.

        :return: An ImageFeature object containing the keypoints and descriptors of the image.
        """
        if image is None:
            raise ValueError("Image provided is None.")
        keypoints, descriptors = self.extractor.detectAndCompute(image, None)

        return ImageFeature(keypoints=keypoints, descriptors=descriptors)


   
