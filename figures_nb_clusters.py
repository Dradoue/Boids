import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    mode = 1

    if mode == 0:

        df0 = pd.read_csv("results/evolution_clusters_statistics_on_1000.csv")
        df1 = pd.read_csv("results/evolution_clusters_statistics_on_500.csv")
        df2 = pd.read_csv("results/evolution_clusters_statistics_on_200.csv")

        df0 = df0.rename(columns={"mean number clusters": "1000 Boids"})
        df1 = df1.rename(columns={"mean number clusters": "500 Boids"})
        df2 = df2.rename(columns={"mean number clusters": "200 Boids"})

        ax = df2.plot(x="time step", y="200 Boids", yerr="std number clusters",
                      legend="200 Boids", figsize=(20, 20))

        ax.set_ylabel("Nombre moyen de clusters avec écart type", fontsize=16)
        ax.set_title(fontsize=16, label="Évolution du nombre de clusters en fonction des populations")

        df1.plot(x="time step", y="500 Boids", yerr="std number clusters", ax=ax)
        df0.plot(x="time step", y="1000 Boids", yerr="std number clusters", ax=ax)
        plt.show()

    elif mode != 0:

        name = "DBSCAN_position_velocities_multistep_Euclidean"
        n_boids = 200
        alpha_ = 1  # [0.6, 0.8, 1, 1.2, 1.4]
        phi_ = 10  # [10, 20, 30]
        gamma_ = 0.9  # [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

        alpha_phi_gamma = str(alpha_) + "_" + str(phi_) + "_" + str(gamma_)
        name_pandas_file = "evolution_ARI_on_" + str(n_boids) + "_" + name + "_" + alpha_phi_gamma
        column_names = ["time step", "num simulation", "ARI"]
        results = pd.DataFrame(columns=column_names)

        name_pandas_file_statistics = "evolution_ARI_statistics_on_" + str(n_boids) + "_" + name \
                                      + "_" + alpha_phi_gamma
        df0 = pd.read_csv("evolution_clusters_statistics_on_1000.csv")
        df1 = pd.read_csv("evolution_clusters_statistics_on_500.csv")
        df2 = pd.read_csv("evolution_clusters_statistics_on_200.csv")

        df0 = df0.rename(columns={"mean number clusters": "1000 Boids"})
        df1 = df1.rename(columns={"mean number clusters": "500 Boids"})
        df2 = df2.rename(columns={"mean number clusters": "200 Boids"})

        ax = df2.plot(x="time step", y="200 Boids", yerr="std number clusters",
                      legend="200 Boids", figsize=(20, 20))

        ax.set_ylabel("Nombre moyen de clusters avec écart type", fontsize=16)
        ax.set_title(fontsize=16, label="Évolution du nombre de clusters en fonction des populations")

        df1.plot(x="time step", y="500 Boids", yerr="std number clusters", ax=ax)
        df0.plot(x="time step", y="1000 Boids", yerr="std number clusters", ax=ax)
        plt.show()
