import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

if __name__ == "__main__":

    # collect the data you want to plot
    # choose a method for the plot
    method = 3
    if method == 1:
        name = "DBSCAN_position_Euclidean_metric"
        eps = [70, 75, 80, 85]
        # min_sample = [2, 4, 6, 8, 10]
        min_sample = [2]

    elif method == 2:
        name = "DBSCAN_position_velocities_Euclidean_metric"
        # param_to_test
        alpha = [0.8, 1, 1.2, 1.2]
        beta = [5, 10, 20, 30, 40, 50, 60]

    elif method == 3:
        name = "DBSCAN_position_velocities_multistep_Euclidean"
        alpha = [0.6, 0.8, 1,  1.2, 1.4]
        phi = [10, 20, 30, 40, 50]
        gamma = [0.95, 0.99]

    elif method == 4:
        name = "DBSCAN_position_velocities_custom_metric"
        alpha = [0.8, 1, 1.2, 1.4]
        phi = [10, 20, 30, 40, 50]

    # choose n_boids to test
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

    # get statistics on number of clusters on time-step :step_to_analyse:
    for n_boids in [200]:  # list_n_boids[:]:

        if n_boids == 200:
            step_to_analyse = step_to_analyse_pop200

        elif n_boids == 500:
            step_to_analyse = step_to_analyse_pop500

        elif n_boids == 1000:
            step_to_analyse = step_to_analyse_pop500

        if method == 1:

            # get the file names related to parameters specified
            all_names = []
            for min_sample_ in min_sample:

                for eps_ in eps:
                    params = "min_sample=" + str(min_sample_) + "_" + "epsilon=" + str(eps_)
                    name_pandas_file_statistics = "evolution_ARI_statistics_on_" + str(
                        n_boids) + "_" + name + "_" + params
                    all_names.append(name_pandas_file_statistics)

            print(all_names)
            # get the data and concat
            if len(all_names) > 1:

                df = pd.read_csv(all_names[0] + ".csv").to_numpy()
                df2 = pd.read_csv(all_names[1] + ".csv").to_numpy()

                df_ = np.concatenate((df[None, :, :], df2[None, :, :]), axis=0)

                for i in np.arange(2, len(all_names)):
                    df = pd.read_csv(all_names[i] + ".csv").to_numpy()
                    df__ = np.concatenate((df_, df[None, :, :]), axis=0)
                    df_ = df__

            label_vec = ['eps=70', 'eps=75', 'eps=80', 'eps=85']
            # label_vec = ['min_sample=2', 'min_sample=3','min_sample=4','min_sample=5']
            title = "evolution of ARI against epsilon parameter"
            # title = "evolution of ARI against min_sample parameter"

            params = "min_sample=" + str(min_sample) + "_" + "epsilon=" + str(eps)
            name_pandas_file_statistics = "evolution_ARI_statistics_on_" + str(
                n_boids) + "_" + name + "_" + params

            data = df_
            N = data.shape[2]
            n_vec = data.shape[0]
            mean = data[:, :, 2]


        if method == 2:

            # get the file names related to parameters specified
            all_names = []
            label_vec = []
            for alpha_ in alpha:

                for beta_ in beta:
                    label_vec.append("alpha=" + str(alpha_) + "_beta=" + str(beta_))
                    params = "alpha=" + str(alpha_) + "_beta=" + str(beta_)
                    name_pandas_file_statistics = "evolution_ARI_statistics_on_" + str(
                        n_boids) + "_" + name + "_" + params
                    all_names.append(name_pandas_file_statistics)

            print(all_names)
            # get the data and concat
            if len(all_names) > 1:

                df = pd.read_csv(all_names[0] + ".csv").to_numpy()
                df2 = pd.read_csv(all_names[1] + ".csv").to_numpy()

                df_ = np.concatenate((df[None, :, :], df2[None, :, :]), axis=0)

                for i in np.arange(2, len(all_names)):
                    df = pd.read_csv(all_names[i] + ".csv").to_numpy()
                    df__ = np.concatenate((df_, df[None, :, :]), axis=0)
                    df_ = df__

            # label_vec = ['min_sample=2', 'min_sample=3','min_sample=4','min_sample=5']
            title = "evolution of ARI against alpha parameter"
            # title = "evolution of ARI against min_sample parameter"

            params = "alpha=" + str(alpha) + "_" + "beta=" + str(beta)
            name_pandas_file_statistics = "evolution_ARI_statistics_on_" + str(
                n_boids) + "_" + name + "_" + params

            data = df_
            N = len(all_names)
            n_vec = data.shape[0]
            mean = data[:, :, 2]
            err_vec = data[:, :, 1]
            x_vec = data[0, :, 0]

        if method == 3:

            # get the file names related to parameters specified
            all_names = []
            label_vec = []
            for alpha_ in alpha:

                for phi_ in phi:

                    for gamma_ in gamma:
                        params = "alpha=" + str(alpha_) + "_" + "phi=" + str(phi_) + "_" + "gamma=" + str(gamma_)
                        label_vec.append(params)
                        name_pandas_file_statistics = "evolution_ARI_statistics_on_" + str(
                            n_boids) + "_" + name + "_" + params
                        all_names.append(name_pandas_file_statistics)

            print(all_names)
            # get the data and concat
            if len(all_names) > 1:

                df = pd.read_csv(all_names[0] + ".csv").to_numpy()
                df2 = pd.read_csv(all_names[1] + ".csv").to_numpy()

                df_ = np.concatenate((df[None, :, :], df2[None, :, :]), axis=0)

                for i in np.arange(2, len(all_names)):
                    df = pd.read_csv(all_names[i] + ".csv").to_numpy()
                    df__ = np.concatenate((df_, df[None, :, :]), axis=0)
                    df_ = df__

            # label_vec = ['alpha=0.8', 'alpha=1', 'alpha=1.2', 'alpha=1.4']
            # label_vec = ['min_sample=2', 'min_sample=3','min_sample=4','min_sample=5']
            title = "evolution of ARI with gamma parameters"
            # title = "evolution of ARI against min_sample parameter"

            params = "alpha=" + str(alpha) + "_" + "phi=" + str(phi) + "_" + "gamma=" + str(gamma)
            name_pandas_file_statistics = "evolution_ARI_statistics_on_" + str(
                n_boids) + "_" + name + "_" + params

            data = df_
            print(data)
            print(data.shape)
            N = len(all_names)
            n_vec = data.shape[0]
            mean = data[:, :, 2]
            err_vec = data[:, :, 1]
            x_vec = data[0, :, 0]

            data_sorted = data.sort_values(["mean ARI", "std ARI", "x"], ascending=False)
