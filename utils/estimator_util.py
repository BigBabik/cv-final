import numpy as np
import cv2


class RANSACConfig:
    def __init__(self, max_iters=1000, threshold=1.0, confidence=0.99):
        self.max_iters = max_iters
        self.threshold = threshold
        self.confidence = confidence

class GCRANSACCOnfig:
    def __init__(self, max_iters=1000, threshold=1.0, confidence=0.99, max_local_optim=100, max_greedy=100):
        self.max_iters = max_iters
        self.threshold = threshold
        self.confidence = confidence
        self.max_local_optim = max_local_optim
        self.max_greedy = max_greedy

class EstimatorConfig:
    def __init__(self, algorithm='RANSAC', **kwargs):
        self.algorithm = algorithm
        if algorithm == 'RANSAC':
            self.params = RANSACConfig(**kwargs)
        elif algorithm == 'GC-RANSAC':
            self.params = GCRANSACCOnfig(**kwargs)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

class FundamentalMatrixEstimator:
    def __init__(self, keypoints1, keypoints2, config:EstimatorConfig = None):
        self.keypoints1 = np.array(keypoints1)
        self.keypoints2 = np.array(keypoints2)
        self.fundamental_matrix = None
        self.config = config


    def estimate(self):
        if len(self.keypoints1) < 8 or len(self.keypoints2) < 8:
            raise ValueError("At least 8 keypoints are required to estimate the fundamental matrix.")
        
        self.fundamental_matrix, mask = cv2.findFundamentalMat(
            self.keypoints1, self.keypoints2, cv2.FM_RANSAC,
            ransacReprojThreshold=self.config.threshold,
            confidence=self.config.confidence,
            maxIters=self.config.max_iters
        )
        return self.fundamental_matrix

    def get_inliers(self):
        raise NotImplementedError("Method not implemented yet.")
    

    
    
    