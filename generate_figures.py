import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ML import DBscan_step_positions, DBscan_step_positions_and_velocity

if __name__ == "__main__":

    list_n_boids = [30, 100, 200, 500, 1000]
    step_to_analyse = 1000
    # calcule le nombre de clusters pour le pas de temps 300 sur chaque dataset.

    name_pandas_file = "nb_clusters_on_" + str(list_n_boids) + "_population_step_" + str(step_to_analyse)
    column_names = ["population per species", "num simulation", "number of clusters"]
    results = pd.DataFrame(columns=column_names)

    name_pandas_file_statistics = "nb_clusters_statistics_on_" + str(list_n_boids) + "_population_step_" + \
                                  str(step_to_analyse)
    column_names = ["population per species", "mean number clusters", "std number clusters"]
    results_statistics = pd.DataFrame(columns=column_names)

    # get statistics on number of clusters on time-step :step_to_analyse:

    for n_boids in list_n_boids:

        num_clusters = []
        for i in range(5):  # 5 different simulations

            if n_boids == 1000 and i == 3:
                break
            # get the directory
            directory_name = "simulation_data_" + str(n_boids) + "_Boids_" + str(i) + "/"
            labels = DBscan_step_positions(step=step_to_analyse, old_labels=None, repository=directory_name,
                                           )
            nb_clusters = np.unique(labels).shape[0] - 1
            print("num clusters time step {0} is {1}".format(step_to_analyse, nb_clusters))

            res = [n_boids, i, nb_clusters]
            results = results.append({"population per species": n_boids, "num simulation": i,
                                      "number of clusters": nb_clusters}, ignore_index=True)

            num_clusters.append(nb_clusters)

        mean_clusters = np.mean(num_clusters)
        std_clusters = np.std(num_clusters)
        results_statistics = results_statistics.append({"population per species": n_boids, "mean number clusters":
            mean_clusters,
                                                        "std number clusters": std_clusters}, ignore_index=True)

    results_statistics.to_csv(name_pandas_file_statistics + ".csv", index=False)
    results.to_csv(name_pandas_file + ".csv", index=False)

    # proceed results
    # produce figures
    fig, ax = plt.subplots()
    results_statistics.plot(x='population per species', y='mean number clusters', yerr='std number clusters', ax=ax)
    plt.show()

if __name__ == "__main__":

    # another part where we watch evolution of number of clusters during the simulation depending on number of boids
    list_n_boids = [30, 100, 200, 500, 1000]
    step_to_analyse = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    # get statistics on number of clusters on time-step :step_to_analyse:
    for n_boids in list_n_boids:

        # create a dataframe for a population
        name_pandas_file = "evolution_clusters_on_" + str(n_boids)
        column_names = ["time step", "num simulation", "number clusters"]
        results = pd.DataFrame(columns=column_names)

        name_pandas_file_statistics = "evolution_clusters_statistics_on_" + str(n_boids)
        column_names = ["time step", "std number clusters", "mean number clusters"]
        results_statistics = pd.DataFrame(columns=column_names)

        for step in step_to_analyse:

            num_clusters = []
            for i in range(5):  # 5 different simulations

                if n_boids == 1000 and i == 3:
                    break
                # get the directory
                directory_name = "simulation_data_" + str(n_boids) + "_Boids_" + str(i) + "/"
                labels = DBscan_step_positions_and_velocity(step=step, old_labels=None, repository=directory_name)
                nb_clusters = np.unique(labels).shape[0] - 1
                print("num clusters time step {0} is {1}".format(step, nb_clusters))

                res = [step, i, nb_clusters]
                results = results.append({"time step": step, "num simulation": i,
                                          "number of clusters": nb_clusters}, ignore_index=True)

                num_clusters.append(nb_clusters)

            mean_clusters = np.mean(num_clusters)
            std_clusters = np.std(num_clusters)

            results_statistics = results_statistics.append({"time step": step, "mean number clusters":
                mean_clusters,
                                                            "std number clusters": std_clusters}, ignore_index=True)

        results_statistics.to_csv(name_pandas_file_statistics + ".csv", index=False)
        results.to_csv(name_pandas_file + ".csv", index=False)

        fig, ax = plt.subplots()
        results_statistics.plot(x='time step', y='mean number clusters', yerr='std number clusters',
                                title="Evolution of clusters for {0} boids".format(str(n_boids)))
        plt.show()
    # proceed results
    # produce figures
