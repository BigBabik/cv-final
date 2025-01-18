import numpy as np
import cv2 as cv


class RANSACConfig:
    def __init__(self, max_iters=1000, threshold=1.0, confidence=0.999999):
        self.max_iters = max_iters # Maximum number of iterations to sample 8 valid points
        self.threshold = threshold 
        self.confidence = confidence 

class GCRANSACCOnfig:
    def __init__(self, max_iters=1000, threshold=0.5, confidence=0.999999, max_local_optim=100, max_greedy=100):
        self.max_iters = max_iters # Maximum number of iterations to sample 8 valid points
        self.threshold = threshold
        self.confidence = confidence
        self.max_local_optim = max_local_optim
        self.max_greedy = max_greedy

class MAGSACConfig:
    def __init__(self, max_iters=10000, threshold=1.25, confidence=0.999999, sigma=1.0):
        self.max_iters = max_iters # Maximum number of iterations to sample 8 valid points
        self.threshold = threshold
        self.confidence = confidence
        self.sigma = sigma

class EstimatorConfig:
    def __init__(self, algorithm='RANSAC',RANSACConfig=None, GCRANSACCOnfig=None):
        self.algorithm = algorithm
        if algorithm == 'RANSAC':
            self.estimator_config = RANSACConfig
        elif algorithm == 'GC-RANSAC':
            self.estimator_config = GCRANSACCOnfig
        elif algorithm == 'MAGSAC':
            self.estimator_config = MAGSACConfig()
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
                self.keypoints1, self.keypoints2, cv.GC_RANSAC,
                ransacReprojThreshold=self.config.estimator_config.threshold,
                confidence=self.config.estimator_config.confidence,
                maxIters=self.config.estimator_config.max_iters
            )
            # Additional GC-RANSAC specific steps can be added here
        elif self.config.algorithm == 'MAGSAC':
            self.fundamental_matrix, self.mask = cv.findFundamentalMat(
            self.keypoints1, self.keypoints2, cv.USAC_MAGSAC,
            ransacReprojThreshold=self.config.estimator_config.threshold,
            confidence=self.config.estimator_config.confidence,
            maxIters=self.config.estimator_config.max_iters
            )
            # Additional MAGSAC specific steps can be added here

        else:
            raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")
        
        if self.fundamental_matrix is None:
            print("[-]Error: Unable to predict fundemental matrix, filling in random value!!")
            self.fundamental_matrix = np.random.rand(9)
        
        self.fundamental_matrix = self.fundamental_matrix.reshape(-1) # flatten it
        return self.fundamental_matrix, self.mask