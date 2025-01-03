General
# what can I offload to the GPU

Loading and preproc
# wrapper functions for preprocessing the dataset

Extraction
# FeatureExtractorConfig to have a general parameter for the config and somehow indicate what algo is is so I can assign blindly,
# regardless of the actual config type

Matching
# How can I filter matches from BFMatcher.match, if I want cross_check?

Estimation
# why are there duplicate inliers in get_inliers? If I cast to a set I drop some points.
# document the estimation configs and function to make sure I understand the parameters I'm feeding in 
