General
# what can I offload to the GPU
# mundane functions to some general utils module
# decouple preproc and sampling for run
# per scene slice of tesst samples in test modes that will resolve the downstream hacks I madein branch testruns, commit deaa1fffa39f28bdc30b13874ff22fc6cda53f33
# add tqdm for the big loops

Loading and preproc
# index_col and set_index dup in loading calibration and coves

Extraction
# FeatureExtractorConfig to have a general parameter for the config and somehow indicate what algo is is so I can assign blindly,
# regardless of the actual config type


Matching
# How can I filter matches from BFMatcher.match, if I want cross_check?
# Batch update for the valid keypoints I save wile matching
# check for failed keypoint1/2 saves to the covis df. RAISE NOTICE ON NO VALID KEYPOINTS FOR A PAIR


Estimation
# document the estimation configs and function to make sure I understand the parameters I'm feeding in 
# dummy values for init or enable None for kp1,kp2
# normalize points before estimation using the calibration data
# understand why the evaluation DF has two T columns... For a patch I use T.iloc[0] to get only the first
