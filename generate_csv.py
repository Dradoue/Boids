from typing import List, Any, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from ML import DBscan_step_positions, DBscan_step_positions_and_velocity, build_ground_truth, \
    DBscan_step_intuition_dist_multistep_1, DBscan_step_intuition_dist
from utils import charge_labels_simulation

if __name__ == "__main__":

    mode = 2
    # 0: analyse number of clusters on list_nb_boids on a particular timestep
    # 1: analyse number of clusters for each population in list_nb_boids on different timesteps
    # 2: analyse ARI scores for each population in list_nb_boids on different timesteps

    if mode == 1:

        # another part where we watch evolution of number of clusters during the simulation depending on number of boids
        list_n_boids = [200, 500, 1000]
        step_to_analyse = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400,
                           1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800,
                           2900, 2999]

        # get statistics on number of clusters on time-step :step_to_analyse:
        for n_boids in list_n_boids:
            list_num_boids = [n_boids] * 4

            # create a dataframe for a population
            name_pandas_file_statistics = "evolution_clusters_statistics_on_" + str(n_boids)
            column_names = ["time step", "std number clusters", "mean number clusters"]
            column_names_p = ["time step", "std number clusters p", "mean number clusters p"]
            column_names_pv = ["time step", "std number clusters pv", "mean number clusters pv"]
            results_statistics = pd.DataFrame(columns=column_names)
            results_statistics_p = pd.DataFrame(columns=column_names_p)
            results_statistics_pv = pd.DataFrame(columns=column_names_pv)

            for step in step_to_analyse:

                num_clusters = []
                num_clusters_pv = []
                num_clusters_p = []
                for i in range(10):  # 10 different simulations

                    # get the directory
                    directory_name = "simulation_data_" + str(list_num_boids) + "_Boids_" + str(i) + "/"

                    labels_pv = DBscan_step_positions_and_velocity(step=step, old_labels=None,
                                                                   repository=directory_name)
                    labels_p = DBscan_step_positions(step=step, old_labels=None, repository=directory_name)

                    labels = build_ground_truth(step=step, old_labels=None, repository=directory_name,
                                                list_nb_boids=list_num_boids)

                    nb_clusters = np.unique(labels).shape[0] - 1
                    nb_clusters_p = np.unique(labels).shape[0] - 1
                    nb_clusters_pv = np.unique(labels).shape[0] - 1
                    print("num clusters time step {0} is {1}".format(step, nb_clusters))
                    num_clusters.append(nb_clusters)
                    num_clusters_pv.append(nb_clusters_pv)
                    num_clusters_p.append(nb_clusters_p)

                mean_clusters = np.mean(num_clusters)
                std_clusters = np.std(num_clusters)
                mean_clusters_p = np.mean(num_clusters_p)
                std_clusters_p = np.std(num_clusters_p)
                mean_clusters_pv = np.mean(num_clusters_pv)
                std_clusters_pv = np.std(num_clusters_pv)

                results_statistics = results_statistics.append({"time step": step, "mean number clusters":
                    mean_clusters,
                                                                "std number clusters": std_clusters}, ignore_index=True)

                results_statistics_p = results_statistics_p.append({"time step": step, "mean number clusters p":
                    mean_clusters_p,
                                                                    "std number clusters p": std_clusters_p},
                                                                   ignore_index=True)
                results_statistics_pv = results_statistics_pv.append({"time step": step, "mean number clusters pv":
                    mean_clusters_pv,
                                                                      "std number clusters pv": std_clusters_pv},
                                                                     ignore_index=True)

            results_statistics.to_csv(name_pandas_file_statistics + ".csv", index=False)

    if mode == 2:

        # another part where we watch evolution of number of clusters during the simulation depending on number of boids
        list_n_boids = [200, 500, 1000]

        # todo to adjust
        step_to_analyse_pop200 = [500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400,
                                  1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800,
                                  2900, 2999]
        step_to_analyse_pop500 = [500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400,
                                  1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800,
                                  2900, 2999]
        step_to_analyse_pop1000 = [500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400,
                                   1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800,
                                   2900, 2999]

        method = 4  # see methods behind

        # get statistics on number of clusters on time-step :step_to_analyse:
        for n_boids in [500, 1000]:  # list_n_boids:

            if n_boids == 200:
                step_to_analyse = step_to_analyse_pop200

            elif n_boids == 500:
                step_to_analyse = step_to_analyse_pop500

            elif n_boids == 1000:
                step_to_analyse = step_to_analyse_pop500

            list_num_boids = [n_boids] * 4
            # create a dataframe for a population
            name = None

            if method == 1:
                name = "DBSCAN_position_Euclidean_metric"
                eps = [70, 75, 80, 85]
                min_sample = [2, 4, 6, 8, 10]

            elif method == 2:
                name = "DBSCAN_position_velocities_Euclidean_metric"
                # param_to_test
                alpha = [0.8, 1, 1.2, 1.4]
                beta = [5, 10, 20, 30, 40, 50, 60]

            elif method == 3:
                name = "DBSCAN_position_velocities_custom_metric"
                alpha = [0.8, 1, 1.2, 1.4]
                phi = [10, 20, 30, 40, 50]

            elif method == 4:
                name = "DBSCAN_position_velocities_multistep_Euclidean"
                alpha = [0.6, 0.8, 1, 1.2, 1.4]
                phi = [10, 20, 30, 40, 50]
                gamma = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

            if method == 1:

                for epsilon in eps:

                    for min_sample_ in min_sample:

                        params = "min_sample=" + str(min_sample_) + "_" + "epsilon=" + str(epsilon)
                        name_pandas_file_statistics = "evolution_ARI_statistics_on_" + str(
                            n_boids) + "_" + name + "_" + params
                        column_names = ["time step", "std ARI", "mean ARI"]
                        results_statistics = pd.DataFrame(columns=column_names)

                        for step in step_to_analyse:

                            ARI_list = []
                            for i in range(10):  # 5 different simulations

                                # get the directory
                                directory_name = "simulation_data_" + str(list_num_boids) + "_Boids_" + str(i) + "/"

                                # get ground truth
                                labels_truth = build_ground_truth(step=step, old_labels=None,
                                                                  repository=directory_name,
                                                                  list_nb_boids=list_num_boids)

                                labels = DBscan_step_positions(step=step, old_labels=None,
                                                               repository=directory_name,
                                                               eps=epsilon, min_sample=min_sample_)

                                ARI = adjusted_rand_score(labels_truth, labels)
                                print("rand score time step {0} is {1}".format(step, ARI))
                                ARI_list.append(ARI)

                            mean_ARI = np.mean(ARI_list)
                            std_ARI = np.std(ARI_list)

                            results_statistics = results_statistics.append({"time step": step, "mean ARI":
                                mean_ARI,
                                                                            "std ARI": std_ARI},
                                                                           ignore_index=True)

                        results_statistics.to_csv(name_pandas_file_statistics + ".csv", index=False)

            if method == 2:

                for alpha_ in alpha:

                    for beta_ in beta:

                        params = "alpha=" + str(alpha_) + "_" + "beta=" + str(beta_)
                        name_pandas_file_statistics = "evolution_ARI_statistics_on_" + str(
                            n_boids) + "_" + name + "_" + params
                        column_names = ["time step", "std ARI", "mean ARI"]
                        results_statistics = pd.DataFrame(columns=column_names)

                        for step in step_to_analyse:

                            ARI_list = []
                            for i in range(10):  # 5 different simulations

                                # get the directory
                                directory_name = "simulation_data_" + str(list_num_boids) + "_Boids_" + str(i) + "/"

                                # get ground truth
                                labels_truth = build_ground_truth(step=step, old_labels=None,
                                                                  repository=directory_name,
                                                                  list_nb_boids=list_num_boids)

                                labels = DBscan_step_positions_and_velocity(step=step, old_labels=None,
                                                                            repository=directory_name,
                                                                            alpha=alpha_,
                                                                            beta=beta_)

                                ARI = adjusted_rand_score(labels_truth, labels)
                                print("rand score time step {0} is {1}".format(step, ARI))
                                ARI_list.append(ARI)

                            mean_ARI = np.mean(ARI_list)
                            std_ARI = np.std(ARI_list)

                            results_statistics = results_statistics.append({"time step": step, "mean ARI":
                                mean_ARI,
                                                                            "std ARI": std_ARI},
                                                                           ignore_index=True)

                        results_statistics.to_csv(name_pandas_file_statistics + ".csv", index=False)

            if method == 3:

                for alpha_ in alpha:

                    for phi_ in phi:

                        params = "alpha=" + str(alpha_) + "_" + "phi=" + str(phi_)
                        name_pandas_file_statistics = "evolution_ARI_statistics_on_" + str(
                            n_boids) + "_" + name + "_" + params
                        column_names = ["time step", "std ARI", "mean ARI"]
                        results_statistics = pd.DataFrame(columns=column_names)

                        for step in step_to_analyse:

                            ARI_list = []
                            for i in range(10):  # 5 different simulations

                                # get the directory
                                directory_name = "simulation_data_" + str(list_num_boids) + "_Boids_" + str(i) + "/"

                                # get ground truth
                                labels_truth = build_ground_truth(step=step, old_labels=None,
                                                                  repository=directory_name,
                                                                  list_nb_boids=list_num_boids)

                                labels = DBscan_step_intuition_dist(step=step, old_labels=None,
                                                                    repository=directory_name,
                                                                    alpha_=alpha_,
                                                                    phi_=phi_)

                                ARI = adjusted_rand_score(labels_truth, labels)
                                print("rand score time step {0} is {1}".format(step, ARI))
                                ARI_list.append(ARI)

                            mean_ARI = np.mean(ARI_list)
                            std_ARI = np.std(ARI_list)

                            results_statistics = results_statistics.append({"time step": step, "mean ARI":
                                mean_ARI,
                                                                            "std ARI": std_ARI},
                                                                           ignore_index=True)

                        results_statistics.to_csv(name_pandas_file_statistics + ".csv", index=False)

            if method == 4:

                for gamma_ in gamma:

                    for phi_ in phi:

                        for alpha_ in alpha:

                            params = "alpha=" + str(alpha_) + "_" + "phi=" + str(phi_) + "_" + "gamma=" + str(gamma_)
                            name_pandas_file_statistics = "evolution_ARI_statistics_on_" + str(n_boids) + "_" + name \
                                                          + "_" + params
                            column_names = ["time step", "std ARI", "mean ARI"]
                            results_statistics = pd.DataFrame(columns=column_names)

                            for step in step_to_analyse:

                                ARI_list = []
                                for i in range(10):  # 5 different simulations

                                    # get the directory
                                    directory_name = "simulation_data_" + str(list_num_boids) + "_Boids_" + str(i) + "/"

                                    # get ground truth
                                    labels_truth = build_ground_truth(step=step, old_labels=None,
                                                                      repository=directory_name,
                                                                      list_nb_boids=list_num_boids)

                                    labels = DBscan_step_intuition_dist_multistep_1(step=step, old_labels=None,
                                                                                    repository=directory_name,
                                                                                    phi_=phi_,
                                                                                    alpha_=alpha_,
                                                                                    gamma_=gamma_)

                                    ARI = adjusted_rand_score(labels_truth, labels)
                                    print("rand score time step {0} is {1}".format(step, ARI))
                                    ARI_list.append(ARI)

                                mean_ARI = np.mean(ARI_list)
                                std_ARI = np.std(ARI_list)

                                results_statistics = results_statistics.append({"time step": step, "mean ARI":
                                    mean_ARI,
                                                                                "std ARI": std_ARI},
                                                                               ignore_index=True)

                            results_statistics.to_csv(name_pandas_file_statistics + ".csv", index=False)
