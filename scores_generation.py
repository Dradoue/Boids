from ML import calculate_rand_score
import numpy as np

if __name__ == "__main__":

    repository = "simulation_data/"
    steps = list(np.arange(300, 500))
    filename_true = "ground_truth_label"
    filename_pred = "DBSCAN_position|velocity_label"

    calculate_rand_score(steps, repository, filename_true, filename_pred)

