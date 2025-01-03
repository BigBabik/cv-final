import numpy as np
import cv2 as cv


class RANSACConfig:
    def __init__(self, max_iters=1000, threshold=1.0, confidence=0.99):
        self.max_iters = max_iters # Maximum number of iterations to sample 8 valid points
        self.threshold = threshold 
        self.confidence = confidence 

class GCRANSACCOnfig:
    def __init__(self, max_iters=1000, threshold=1.0, confidence=0.99, max_local_optim=100, max_greedy=100):
        self.max_iters = max_iters # Maximum number of iterations to sample 8 valid points
        self.threshold = threshold
        self.confidence = confidence
        self.max_local_optim = max_local_optim
        self.max_greedy = max_greedy

class EstimatorConfig:
    def __init__(self, algorithm='RANSAC',RANSACConfig=None, GCRANSACCOnfig=None):
        self.algorithm = algorithm
        if algorithm == 'RANSAC':
            self.estimator_config = RANSACConfig
        elif algorithm == 'GC-RANSAC':
            self.estimator_config = GCRANSACCOnfig
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

class FundamentalMatrixEstimator:
    def __init__(self, keypoints1, keypoints2, config:EstimatorConfig = None):
        if type(keypoints1[0]) == cv.KeyPoint:
            keypoints1 = [kp.pt for kp in keypoints1]
            keypoints2 = [kp.pt for kp in keypoints2]

        self.keypoints1 = np.array(keypoints1)
        self.keypoints2 = np.array(keypoints2)

        self.fundamental_matrix = None
        self.mask = None
        self.config = config

    def estimate(self):
        if len(self.keypoints1) < 8 or len(self.keypoints2) < 8:
            raise ValueError("At least 8 keypoints are required to estimate the fundamental matrix.")
        
        if self.config.algorithm == 'RANSAC':
            self.fundamental_matrix, self.mask = cv.findFundamentalMat(
                self.keypoints1, self.keypoints2, cv.FM_RANSAC,
                ransacReprojThreshold=self.config.estimator_config.threshold,
                confidence=self.config.estimator_config.confidence,
                maxIters=self.config.estimator_config.max_iters
            )

        elif self.config.algorithm == 'GC-RANSAC':
            self.fundamental_matrix, self.mask = cv.findFundamentalMat(
                self.keypoints1, self.keypoints2, cv.FM_RANSAC,
                ransacReprojThreshold=self.config.estimator_config.threshold,
                confidence=self.config.estimator_config.confidence,
                maxIters=self.config.estimator_config.max_iters
            )
            # Additional GC-RANSAC specific steps can be added here

        else:
            raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")
        
        return self.fundamental_matrix

    def get_inliers(self):
        if self.fundamental_matrix is None or self.mask is None:
            raise ValueError("Fundamental matrix has not been estimated yet.")
        
        inliers1 = self.keypoints1[self.mask.ravel() == 1]
        inliers2 = self.keypoints2[self.mask.ravel() == 1]
        
        return inliers1, inliers2
   