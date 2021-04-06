from ML import calculate_rand_score
import numpy as np

if __name__ == "__main__":

    repository = "simulation_data_2/"  # repository where the data is (in data/*repository*)
    steps = list(np.arange(300, 500))  # steps to take into account in the calculation
    filename_true = "ground_truth_label"  # file name for ground-truth (see for example file_name
    # argument in stock_file function in build_ground_truth function in module ML.py)
    filename_pred = "DBSCAN_position|velocity_label"  # other filename for getting labels

    # calculate mean rand score around steps and pred - true labels and show it
    calculate_rand_score(steps, repository, filename_true, filename_pred)

