import numpy as np

# Function to calculate rotation and translation errors from fundamental matrices.
def compute_pose_errors(F_pred, F_gt):
    # Ensure the inputs are 2D arrays
    F_pred = np.atleast_2d(F_pred)
    F_gt = np.atleast_2d(F_gt)

    # Normalize fundamental matrices
    F_pred = F_pred / np.linalg.norm(F_pred, 'fro')
    F_gt = F_gt / np.linalg.norm(F_gt, 'fro')

    # Compute the difference between the predicted and ground truth fundamental matrices
    F_diff = F_pred - F_gt

    # Calculate rotation error: Frobenius norm of the difference (as a proxy for angular error)
    rotation_error = np.linalg.norm(F_diff, ord='fro')

    # Calculate translation error: Sum of absolute differences between the matrices
    translation_error = np.sum(np.abs(F_diff))

    return rotation_error, translation_error

# Evaluation thresholds
thresholds_r = np.linspace(1, 10, 10)  # Rotation thresholds (degrees)
thresholds_t = np.geomspace(0.2, 5, 10)  # Translation thresholds (meters)

def evaluate_pose_accuracy(F_pred, F_gt):
    rotation_error, translation_error = compute_pose_errors(F_pred, F_gt)
    
    accuracies = []
    for r_thresh, t_thresh in zip(thresholds_r, thresholds_t):
        is_accurate = (rotation_error <= r_thresh) and (translation_error <= t_thresh)
        accuracies.append(is_accurate)

    # Calculate percentage of accurate pairs over all thresholds
    accuracy_percentage = np.mean(accuracies) * 100
    return accuracy_percentage

def mean_average_accuracy(F_list):
    scene_accuracies = []

    for F_pred, F_gt in F_list:
        accuracy = evaluate_pose_accuracy(F_pred, F_gt)
        scene_accuracies.append(accuracy)

    # Average accuracy across all scenes
    return np.mean(scene_accuracies)

def calc_maa(F_pred, F_gt):
    F_list = [
        (F_pred,  F_gt),  # Scene 1: (F_pred, F_gt)
    ]
    mAA = mean_average_accuracy(F_list)
    return mAA


# Test setup
if __name__ == "__main__":
    # Example input: list of fundamental matrices for multiple scenes
    F_list = [
        (np.random.rand(3, 3), np.random.rand(3, 3)),  # Scene 1: (F_pred, F_gt)
        (np.random.rand(3, 3), np.random.rand(3, 3)),  # Scene 2: (F_pred, F_gt)
    ]

    mAA = mean_average_accuracy(F_list)
    print(mAA)
