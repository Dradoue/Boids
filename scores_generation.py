import numpy as np

from ML import calculate_rand_score

if __name__ == "__main__":
    repository = "simulation_data/"  # repository where the data is (in data/*repository*)

    steps = list(np.arange(300, 1000))  # steps to take into account in the calculation

    filename_true = "ground_truth_label"  # file name for ground-truth (see for example file_name
    # argument in stock_file function in build_ground_truth function in module ML.py)

    filename_pred_DBSCAN_using_positions_velocities = "DBSCAN_position|velocity_label"  # other filename for getting labels

    filename_pred_DBSCAN_using_positions = "DBSCAN_positions_label"

    filename_pred_DBSCAN_intuition_dist = "DBSCAN_intuition_dist_phi=100_alpha=1.2_label"

    # calculate mean rand score around steps and pred - true labels and show it
    calculate_rand_score(steps, repository, filename_true, filename_pred_DBSCAN_using_positions_velocities)
    calculate_rand_score(steps, repository, filename_true, filename_pred_DBSCAN_using_positions)
    calculate_rand_score(steps, repository, filename_true, filename_pred_DBSCAN_intuition_dist)
