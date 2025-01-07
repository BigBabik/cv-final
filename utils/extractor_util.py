from dataclasses import dataclass
from typing import Tuple, Optional
import cv2 as cv
import numpy as np

@dataclass
class ImageFeature:
    keypoints: list
    descriptors: Optional[Tuple]

class SIFTConfig:
    nfeatures: int = 0
    contrastThreshold: float = 0.04
    edgeThreshold: float = 10.0
    sigma: float = 1.6

class FeatureExtractorConfig:
    def __init__(self, sift: Optional[SIFTConfig] = None):
        if sift is not None:
            self.sift = sift
        else:
            raise ValueError("No specific feature extractor config specified in general config.")

class FeatureExtractor:
    def __init__(self, config: FeatureExtractorConfig):
        self.config = config
        if self.config.sift is not None: 
            self.sift = cv.SIFT_create(
                nfeatures=self.config.sift.nfeatures,
                contrastThreshold=self.config.sift.contrastThreshold,
                edgeThreshold=self.config.sift.edgeThreshold,
                sigma=self.config.sift.sigma
            )

        else:
            raise ValueError("No feature extractor specified in config.")

    def extract_features(self, image: np.ndarray) -> ImageFeature:    
        """
        Extracts features from an image using the algorithm provided to the extractor.

        :param image: The image to extract features from, as a numpy array from the preprocessing step.

        :return: An ImageFeature object containing the keypoints and descriptors of the image.
        """
        if image is None:
            raise ValueError("Image provided is None.")
        keypoints, descriptors = self.sift.detectAndCompute(image, None)

        return ImageFeature(keypoints=keypoints, descriptors=descriptors)
    
class BFMatcherConfig:
    def __init__(self, norm_type: int = cv.NORM_L2, cross_check: bool = True, knn: bool = False):
        self.norm_type = norm_type
        self.cross_check = cross_check
        self.knn = knn

class FLANNMatcherConfig:
    def __init__(self, algorithm: int = 1, trees: int = 5, checks: int = 50):
        self.algorithm = algorithm
        self.trees = trees
        self.checks = checks

class MatcherConfig:
    def __init__(self, bf: BFMatcherConfig = None, flann: FLANNMatcherConfig = None):
        if bf is not None:
            if bf.knn:
                self.matcher_type = 'BF-KNN'
            else:
                self.matcher_type = 'BF'
                self.matcher_config = bf

        elif flann is not None:
            self.matcher_type = 'FLANN'
            self.matcher_config = flann 
        else:
            raise ValueError("No matcher specified in config.")

class FeatureMatcher:
    def __init__(self, config: MatcherConfig):
        self.config = config
        if self.config.matcher_type == 'BF':
            self.matcher = cv.BFMatcher(crossCheck=self.config.matcher_config.cross_check, normType=self.config.matcher_config.norm_type) 
        elif self.config.matcher_type == 'BF-KNN':
            self.matcher = cv.BFMatcher()
        elif self.config.matcher_type == 'FLANN':
            index_params = dict(algorithm=self.config.flann.algorithm, trees=self.config.flann.trees)
            search_params = dict(checks=self.config.flann.checks)
            self.matcher = cv.FlannBasedMatcher(index_params, search_params)
        else:
            raise ValueError("Unsupported matcher type specified in config.")

    def match_features(self, features1: ImageFeature, features2: ImageFeature):
        """
        Matches features between two images.

        :param features1: ImageFeature object containing keypoints and descriptors of the first image.
        :param features2: ImageFeature object containing keypoints and descriptors of the second image.

        :return: List of matched features.
        """
        if features1.descriptors is None or features2.descriptors is None:
            raise ValueError("Descriptors cannot be None for matching.")
        
        if self.config.matcher_type == 'BF-KNN':
            matches = self.matcher.knnMatch(features1.descriptors, features2.descriptors, k=2)

        elif self.config.matcher_type == 'BF':
            matches = self.matcher.match(features1.descriptors, features2.descriptors)
            matches = sorted(matches, key=lambda x: x.distance)

        return matches
   
    def filter_lowe_matches(self, matches, features1: ImageFeature, features2: ImageFeature, ratio: float = 0.75):
        """
        Filters matches using Lowe's ratio test.

        :param matches: List of DMatch objects.
        :param features1: ImageFeature object containing keypoints and descriptors of the first image.
        :param features2: ImageFeature object containing keypoints and descriptors of the second image.
        :param ratio: Lowe's ratio for filtering matches.

        :return: List of valid matches and corresponding keypoints from both images.
        """
        if self.config.matcher_type != 'BF-KNN':
            raise ValueError("Lowe's ratio test cannot be applied to matches not done by a 2 kernel KNN.")
        
        valid_matches = []
        keypoints1 = []
        keypoints2 = []

        while len(valid_matches) < 8 and ratio < 1.0:
            valid_matches = []
            keypoints1 = []
            keypoints2 = []
            for m, n in matches:
                if m.distance < ratio * n.distance:
                    valid_matches.append(m)
                    keypoints1.append(features1.keypoints[m.queryIdx])
                    keypoints2.append(features2.keypoints[m.trainIdx])

            ratio += 0.05

        if len(valid_matches) < 8:
            print("FAILLLLLLLLLINGGGGG")
        return valid_matches, keypoints1, keypoints2