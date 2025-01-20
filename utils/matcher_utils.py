import cv2 as cv
from enum import Enum
from utils import extractor_util as exu

class MatcherType(Enum):
    BFMatcher = 1
    BFMatcherWithKNN = 2
    FLANN = 3


def _create_BFMatcher(
        norm_type: int = cv.NORM_L2,
        cross_check: bool = True,
        knn: bool = False):
    if knn:
        return cv.BFMatcher()
    
    bfmatcher = cv.BFMatcher(crossCheck=cross_check, normType=norm_type) 
    return bfmatcher

def _create_FLANN_matcher(
        algorithm: int = 1,
        trees: int = 5,
        checks: int = 50):
    
    index_params = dict(algorithm=algorithm, trees=trees)
    search_params = dict(checks=checks)
    flann_matcher = cv.FlannBasedMatcher(index_params, search_params)

    return flann_matcher


class FeatureMatcher:

    def __init__(self, matcher_type: MatcherType):
        self.matcher_type = matcher_type
        match matcher_type:
            case MatcherType.BFMatcher:
                self.matcher = _create_BFMatcher(knn=False)
            case MatcherType.BFMatcherWithKNN:
                self.matcher = _create_BFMatcher(knn=True)
            case MatcherType.FLANN:
                self.matcher = _create_FLANN_matcher()
            case default:
                raise ValueError("Unsupported matcher type specified in config.")


    def match_features(self, features1: exu.ImageFeature, features2: exu.ImageFeature):
        """
        Matches features between two images.

        :param features1: ImageFeature object containing keypoints and descriptors of the first image.
        :param features2: ImageFeature object containing keypoints and descriptors of the second image.

        :return: List of matched features.
        """
        if features1.descriptors is None or features2.descriptors is None:
            raise ValueError("Descriptors cannot be None for matching.")
        
        if self.matcher_type == MatcherType.BFMatcherWithKNN:
            matches = self.matcher.knnMatch(features1.descriptors, features2.descriptors, k=2)

        elif self.matcher_type == MatcherType.BFMatcher:
            matches = self.matcher.match(features1.descriptors, features2.descriptors)
            matches = sorted(matches, key=lambda x: x.distance)

        return matches
   
    def filter_lowe_matches(
            self,
            matches,
            features1: exu.ImageFeature,
            features2: exu.ImageFeature,
            ratio: float = 0.85):
        """
        Filters matches using Lowe's ratio test.

        :param matches: List of DMatch objects.
        :param features1: ImageFeature object containing keypoints and descriptors of the first image.
        :param features2: ImageFeature object containing keypoints and descriptors of the second image.
        :param ratio: Lowe's ratio for filtering matches.

        :return: List of valid matches and corresponding keypoints from both images.
        """
        if self.matcher_type != MatcherType.BFMatcherWithKNN:
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