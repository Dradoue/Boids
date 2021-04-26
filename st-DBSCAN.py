import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.utils import check_array
from ML import calculate_rand_score, stock_labels
from utils import get_dataset_steps_positions_velocities_headings


def get_mean_score(name_file):
    try:
        with open(name_file):

            score = np.mean(np.array(np.loadtxt(name_file), dtype=float))

            return score

    except IOError:
        print("Could not open file {0}".format(name_file))
        exit()


class ST_DBSCAN:
    """
        ST_DBSCAN class for clustering

        ref
        - ST-DBSCAN: An algorithm for clustering spatial–temporal data
        Derya Birant, Alp Kut

        ----------
        :param eps1: float, the density threshold for spatial neighborhood
        :param eps2: float, The temporal threshold for temporal neighborhood
        :param min_samples: The number of samples required for an object to be a core point.
        :param metric_1: string, metric for spatial neighborhood
        :param metric_2: string, metric for temporal neighborhood

        string default='euclidean', can also be a custom function
        The used distance metric - more options are
        ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’,
        ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’,
        ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘rogerstanimoto’, ‘sqeuclidean’,
        ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘yule’.

        :param indices_1: list of column indices where spatial attributes are situated in the data
        :param indices_2: list of column indices where non-spatial attributes are situated in the data
    """

    def __init__(self,
                 eps1,
                 eps2,
                 min_samples,
                 indices_1,
                 indices_2,
                 metric_1='euclidean',
                 metric_2='euclidean',
                 ):

        self.eps1 = eps1
        self.eps2 = eps2
        self.indices_1 = indices_1
        self.indices_2 = indices_2
        self.min_samples = min_samples
        self.metric_1 = metric_1
        self.metric_2 = metric_2
        self.labels = None

        assert self.eps1 > 0, 'eps1 must be positive'
        assert self.eps2 > 0, 'eps2 must be positive'
        assert type(self.min_samples) == int, 'min_samples must be a positive integer'
        assert self.min_samples > 0, 'min_samples must be a positive integer'

    def fit(self, X):

        # check if input is correct
        X = check_array(X)

        if not self.eps1 > 0.0 or not self.eps2 > 0.0 or not self.min_samples > 0.0:
            raise ValueError('eps1, eps2, minPts must be positive')

        # Compute squared distance matrix for
        non_spatial_square_dist_matrix = pdist(X[:, self.indices_1], metric=self.metric_1)

        spatial_square_dist_matrix = pdist(X[:, self.indices_2], metric=self.metric_2)

        # filter the euc_dist matrix using the time_dist
        dist = np.where(non_spatial_square_dist_matrix <= self.eps1, spatial_square_dist_matrix, 2 * self.eps2)

        db = DBSCAN(eps=self.eps2,
                    min_samples=self.min_samples,
                    metric='precomputed')

        db.fit(squareform(dist))

        self.labels = db.labels_

    def stock_labels_to_directory(self, directory, nb_obs, step_init, step_end):

        true_steps = np.arange(step_init, step_end)
        steps = np.arange(0, step_end - step_init)

        ind_init = steps[0]
        # for each step
        for (step, step_true) in zip(steps, true_steps):
            # get all the observations of step i

            ind_to_get = np.arange(ind_init, ind_init + nb_obs)
            label_step = self.labels[ind_to_get]

            stock_labels(label_step, step_true, repository=directory,
                         filename="ST_DBSCAN_eps1=" + str(self.eps1) + "eps2="
                                  + str(self.eps2) + "Nsample="
                                  + str(self.min_samples) + "label")
            ind_init += nb_obs

    def generate_results(self, directory, step_init, step_end):

        steps = list(np.arange(step_init, step_end + 1))  # steps to take into account in the calculation
        filename_true = "ground_truth_label"  # file name for ground-truth (see for example file_name
        # argument in stock_file function in build_ground_truth function in module ML.py)
        filename_pred = "ST_DBSCAN_eps1=" + str(self.eps1) + "eps2=" + \
                        str(self.eps2) + "Nsample=" + str(self.min_samples) + "label"

        score_mean = calculate_rand_score(steps, directory, filename_true, filename_pred)
        return score_mean


def split_data(data, n_indiv, time_step):
    list_data = list()

    for i in np.arange(0, data.shape[0], n_indiv * time_step):
        list_data.append(data[i:i + time_step * n_indiv, :])

    return list_data


if __name__ == "__main__":

    n_indiv = 120
    n_time_step = 3
    directory = "simulation_data/"
    step_init = 300
    step_end = 1000

    # build the dataset
    data = get_dataset_steps_positions_velocities_headings(step_init, step_end, n_indiv, directory)

    # split the dateset to have series of n_time_step times-steps with positions for each boids
    list_data = split_data(data, n_indiv, n_time_step)

    # parameters
    eps1 = [3]
    eps2 = [1, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 85, 90]
    min_samples = [2, 3, 5, 7, 10, 20]
	n_test = 40
    # eps1 = 2 -> eps2=80, min_sample=5 best found
    # eps1 = 1 -> eps2=80, min_sample=5 best found
    # eps1 = 3 -> eps2=80, min_sample=5 best found

    results = list()
    param_list = list()

    # test each parameters
    for eps_1 in eps1:
        for eps_2 in eps2:
            for min_sample in min_samples:

                # we test on n_test sets of n_time_frame samples
                mean_res = list()

                for i in range(n_test):
                    st_dbscan = ST_DBSCAN(eps_1, eps_2, min_sample, [0], [1, 2])
                    st_dbscan.fit(list_data[i])

                    st_dbscan.stock_labels_to_directory(directory=directory,
                                                        nb_obs=n_indiv,
                                                        step_init=step_init + i * n_time_step,
                                                        step_end=step_init + (i + 1) * n_time_step)

                    # generate results will calculate
                    print("eps1: ", eps_1)
                    print("eps2: ", eps_2)
                    print("min_sample", min_sample)

                    ari_score = st_dbscan.generate_results(directory=directory,
                                                           step_init=step_init + i * n_time_step,
                                                           step_end=step_init + (i + 1) * n_time_step - 1)
                    mean_res.append(ari_score)

                results.append(np.mean(mean_res))
                param_list.append([eps_1, eps_2, min_sample])

    print("ind best parameter", np.argmax(results))
    print("best score", results[np.argmax(results)])
    print("best parameters: eps1, eps2, min_sample = ", param_list[np.argmax(results)])
