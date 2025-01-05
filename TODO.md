General
# what can I offload to the GPU

Loading and preproc
# fix the calibration DF while loading the dataset instead of in functions.

Extraction
# FeatureExtractorConfig to have a general parameter for the config and somehow indicate what algo is is so I can assign blindly,
# regardless of the actual config type


Matching
# How can I filter matches from BFMatcher.match, if I want cross_check?
# Batch update for the valid keypoints I save wile matching
# check for failed keypoint1/2 saves to the covis df. RAISE NOTICE ON NO VALID KEYPOINTS FOR A PAIR


Estimation
# why are there duplicate inliers in get_inliers? If I cast to a set I drop some points.
# document the estimation configs and function to make sure I understand the parameters I'm feeding in 
# dummy values for init or enable None for kp1,kp2
# normalize points before estimation using the calibration data
