import collections
import time

import karateclub
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score

from constants import TRIANGLES_COLORS
from utils import toroid_dist_between_points, \
    get_positions_velocity_headings, charge_labels_simulation

global phi
global alpha
phi = 300
alpha = 1.2


def nb_clusters(labels):
    return np.unique(labels).shape[0] - 1


def build_graph(positions, velocities, headings):
    g = nx.Graph()
    data = transform_data_for_graph(positions, velocities, headings)
    g.add_nodes_from(data)
    list_edges = calculate_edges_1(positions, velocities, headings)
    g.add_edges_from(list_edges)
    return g


def transform_data_for_graph(positions, velocities, headings):
    return [(i, dict(pos=tuple(pos), vel=tuple(vel), head=head))
            for i, pos, vel, head in
            zip(list(range(positions.shape[0])), positions.tolist(),
                velocities.tolist(), headings.tolist())]


def calculate_edges_1(positions, velocities, headings, epsilon=150):
    """
    simple method for calculating edges with positions,
    velocities and headings to be used later
    """
    list_edges = []
    for i in range(positions.shape[0]):
        for j in list(range(positions.shape[0])):
            if i != j:
                if toroid_dist_between_points(positions[i, :],
                                              positions[j, :]) < epsilon:
                    list_edges.append((i, j))

    return list_edges


def connected_components_graph(graph):
    """
    find connected components
    return list of graphs
    """
    graphs_ = [graph.subgraph(c).copy() for c
               in nx.connected_components(graph)]

    # rename each subgraph, stock the renaming into a dictionary
    list_renaming = []
    list_new_graphs = []
    for graph in graphs_:
        renaming = dict()
        for i in range(len(graph.nodes)):
            nodes = list(graph.nodes)
            renaming[nodes[i]] = i

        list_new_graphs.append(nx.relabel.relabel_nodes(graph, renaming))
        list_renaming.append(renaming)

    return list_new_graphs, list_renaming


def label_prop(graph):
    """
    label propagation algorithm
    """
    model = karateclub.LabelPropagation()
    model.fit(graph)

    return model.get_memberships()


def modify_colors(triangles, color_list):
    """
    assign color_list to triangles
    """
    triangles.set_colors(color_list)


def membership_to_colorlist(membership):
    cluster_membership = [membership[node] for node in
                          range(len(membership))]

    color_list = []
    for i in range(len(cluster_membership)):
        color_list.append(TRIANGLES_COLORS[cluster_membership[i]
                                           % len(TRIANGLES_COLORS)])
    return color_list


def graph_step(step, repository="simulation_data/"):
    positions, velocities, headings = \
        get_positions_velocity_headings(repository, step)

    graph = build_graph(positions, velocities, headings)

    if len(graph.edges) > 0:

        color_dict = dict()
        list_connected_comp, list_renaming = \
            connected_components_graph(graph)

        for subgraph, renaming in zip(list_connected_comp,
                                      list_renaming):

            inv_ren = {v: k for k, v in renaming.items()}
            # if there is one edge or more
            if len(subgraph.edges) > 0:

                membership = label_prop(subgraph)
                color_list = membership_to_colorlist(membership)
                for i in range(len(membership)):
                    color_dict[inv_ren[i]] = color_list[i]

            # else, imply one node alone
            else:
                color_dict[inv_ren[0]] = TRIANGLES_COLORS[0]

        color_dict = collections.OrderedDict(sorted(color_dict.items()))
        list_color = list(color_dict.values())
        return list_color


############################utils

def merge_labels(old_labels, new_labels):
    n_old_labels = np.unique(old_labels).shape[0]

    for i in range(n_old_labels):

        old_indices_label_i = np.where(old_labels == i)[0]

        unique, counts = np.unique(new_labels[old_indices_label_i],
                                   return_counts=True)

        if counts.shape[0] > 0:
            arg_ind_max = np.argmax(counts)

            if counts[arg_ind_max] > old_indices_label_i.shape[0] // 2:

                to_replace = unique[arg_ind_max]

                if to_replace != i:
                    new_labels = np.where(new_labels == to_replace,
                                          i, new_labels)
                    new_labels = np.where(new_labels == i,
                                          to_replace, new_labels)
    return new_labels


def stock_labels(labels, step, repository, filename):
    # print("data/" + repository + filename + str(step) + "saved")
    np.savetxt("data/" + repository + filename + str(step), labels)


def labels_to_colorlist(labels):
    """
    assign a color to a label for each label in
    the labels list, return a list of colors
    """
    color_list = []
    for i in range(len(labels)):
        color_list.append(TRIANGLES_COLORS[labels[i]])
    return color_list

################################# DBSCAN

def DBscan_step_positions(step, old_labels, repository, eps=85, min_sample=2):
    """
    DBSCAN algorithm on positions
    """
    positions, velocities, headings = \
        get_positions_velocity_headings(repository, step)
    # train_data = np.concatenate((positions, velocities), axis=1)
    train_data = positions
    start = time.time()
    db = DBSCAN(eps=eps, min_samples=min_sample).fit(train_data)
    labels = db.labels_ + 1  # for getting rid of -1 labels
    end = time.time()
    print("clustering done in: {0} seconds".format(end - start))
    if old_labels is not None:
        labels = merge_labels(old_labels, labels)
    stock_labels(labels, step, repository=repository,
                 filename="DBSCAN_positions_eps=" + str(eps) + "min_sample=" + str(min_sample) + "_label")

    return labels


def test_DBSCAN_positions(steps, directory, list_eps, list_min_sample):
    name_pandas_file = "DBSCAN_position_table_results"
    column_names = ["eps", "min sample", "mean ARI score"]

    # create empty dataframe
    results = pd.DataFrame(columns=column_names)

    for eps in list_eps:
        for min_sample in list_min_sample:

            # produce results for DBSCAN algorithm with these values of eps and min_sample

            old_labels = DBscan_step_positions(steps[0], None, directory, eps=eps, min_sample=min_sample)

            for step in steps[1:]:
                old_labels = DBscan_step_positions(step, old_labels, directory, eps=eps, min_sample=min_sample)

            # produce results
            filename_true = "ground_truth_label"
            filename_pred = "DBSCAN_positions_eps=" + str(eps) + "min_sample=" + str(min_sample) + "_label"

            score = calculate_rand_score(steps, directory, filename_true, filename_pred)
            results = results.append({"eps": eps, "min sample": min_sample, "mean ARI score": score}, ignore_index=True)
            print(results)

    # stock dataframe into a file
    results.to_csv(name_pandas_file + ".csv", index=False)


def DBscan_step_positions_and_velocity(step, old_labels, repository,
                                       alpha=1,
                                       beta=27,
                                       eps=85,
                                       min_sample=2):
    """
    DBSCAN algorithm on positions + beta * velocities
    """
    positions, velocities, headings = \
        get_positions_velocity_headings(repository, step)

    train_data = np.concatenate((alpha * positions, beta * velocities), axis=1)

    start = time.time()
    db = DBSCAN(eps=eps, min_samples=min_sample).fit(train_data)
    end = time.time()
    print("clustering done in: {0} seconds".format(end - start))
    labels = db.labels_ + 1  # for getting rid of -1 labels
    if old_labels is not None:
        labels = merge_labels(old_labels, labels)
    stock_labels(labels, step, repository=repository,
                 filename="DBSCAN_position|velocity_eps=" + str(eps) + "min_sample=" + str(min_sample)
                          + "alpha=" + str(alpha) + "beta=" + str(beta) + "label")

    return labels


def test_DBSCAN_positions_and_velocity(steps, directory, list_eps, list_min_sample, list_alpha, list_beta):
    name_pandas_file = "DBSCAN_position_and_velocity_table_results"
    column_names = ["eps", "min sample", "alpha", "beta", "mean ARI score"]

    # create empty dataframe
    results = pd.DataFrame(columns=column_names)

    for eps in list_eps:
        for min_sample in list_min_sample:
            for beta in list_beta:
                for alpha in list_alpha:

                    # produce results for DBSCAN algorithm with these values of eps and min_sample
                    old_labels = DBscan_step_positions_and_velocity(steps[0], None, directory, alpha=alpha,
                                                                    beta=beta,
                                                                    min_sample=min_sample,
                                                                    eps=eps)

                    for step in steps[1:]:
                        old_labels = DBscan_step_positions_and_velocity(step,
                                                                        old_labels,
                                                                        directory,
                                                                        alpha=alpha,
                                                                        beta=beta,
                                                                        min_sample=min_sample,
                                                                        eps=eps)

                    # produce results
                    filename_true = "ground_truth_label"
                    filename_pred = "DBSCAN_position|velocity_eps=" + str(eps) + "min_sample=" + str(min_sample) \
                                    + "alpha=" + str(alpha) + "beta=" + str(beta) + "label"

                    score = calculate_rand_score(steps, directory, filename_true, filename_pred)
                    results = results.append({"eps": eps, "min sample": min_sample, "alpha": alpha, "beta": beta,
                                              "mean ARI score": score}, ignore_index=True)
                    print(results)
    # stock dataframe into a file
    results.to_csv(name_pandas_file + ".csv", index=False)


def linear_comb_dist12(a1, a2):
    global phi, alpha
    return alpha * d1(a1, a2) + phi * d2(a1, a2)


def linear_comb_dist12_multiplestep(a1, a2, nb_step=3, gamma=0.5):
    # a1 and a2 are matrices with multiple timesteps data
    res = 0
    for i in range(0, nb_step):
        res = res + gamma ** (nb_step - i) * linear_comb_dist12(a1[i:i + 2], a2[i:i + 2])
    return res


def d1(a1, a2):
    res = np.linalg.norm(a1[:2] - a2[:2])
    return res


def d2(a1, a2):
    similarity = np.dot(a1[2:], a2[2:]) / (np.linalg.norm(a1[2:]) * np.linalg.norm(a2[2:]))
    angular_dist = np.abs(similarity - 1)
    return angular_dist


def d1(a1, a2):
    res = np.linalg.norm(a1[:2] - a2[:2])
    return res


def linear_comb_dist12_multistep_precomputed(X, nb_step=3, gamma=0.5):
    global phi, alpha

    list_linear_dist = np.array([alpha * pdist(X[:, i:i + 2], metric=d1)
                                 + phi * pdist(X[:, i + 2:i + 4], metric=d1) for i in
                                 range(0, (nb_step - 1) * 4 + 1, 4)])

    list_linear_dist_with_gamma = np.array([gamma ** (nb_step - i) * list_linear_dist[i] for i in range(0, nb_step)])

    final_result = np.sum(list_linear_dist_with_gamma, axis=0)

    return squareform(final_result)


def DBscan_step_intuition_dist_multistep(step, old_labels, repository, min_sample=2,
                                         eps=85, nb_step=3, gamma=0.5):
    """
    DBSCAN algorithm on positions + beta * velocities
    """
    global phi, alpha
    precomputed_ = True
    train_data = None

    for i in range(step, step + nb_step, 1):

        positions, velocities, headings = \
            get_positions_velocity_headings(repository, step)

        if train_data is not None:

            train_data = np.concatenate((train_data, np.concatenate((positions, velocities), axis=1)),
                                        axis=1)
        else:

            train_data = np.concatenate((positions, velocities), axis=1)

    start = time.time()

    if precomputed_:

        train_data = linear_comb_dist12_multistep_precomputed(train_data)
        db = DBSCAN(eps=eps, min_samples=min_sample, metric='precomputed').fit(train_data)

    else:
        db = DBSCAN(eps=eps, min_samples=min_sample, metric=linear_comb_dist12_multiplestep).fit(train_data)

    end = time.time()
    print("clustering done in: {0} seconds".format(end - start))
    labels = db.labels_ + 1  # for getting rid of -1 labels

    if old_labels is not None:
        labels = merge_labels(old_labels, labels)
    stock_labels(labels, step, repository=repository,
                 filename="DBSCAN_intuition_distmultisteps_phi=" + str(phi) + "_alpha="
                          + str(alpha) + "gamma=" + str(gamma) + "_label")

    return labels


def DBscan_step_intuition_dist(step, old_labels, repository,
                               min_sample=2, eps=85):
    """
    DBcsan algorithm on positions + beta * velocities
    """
    global phi, alpha

    positions, velocities, headings = \
        get_positions_velocity_headings(repository, step)

    train_data = np.concatenate((positions, velocities), axis=1)
    start = time.time()

    db = DBSCAN(eps=eps, min_samples=min_sample, metric=linear_comb_dist12).fit(train_data)

    end = time.time()
    print("clustering done in: {0} seconds".format(end - start))

    labels = db.labels_ + 1  # for getting rid of -1 labels
    if old_labels is not None:
        labels = merge_labels(old_labels, labels)
    stock_labels(labels, step, repository=repository,
                 filename="DBSCAN_intuition_dist_phi=" + str(phi) + "_alpha=" + str(alpha) + "_label")
    return labels


def test_DBSCAN_new_metric_positions_and_velocity(steps, directory, list_alpha, list_phi):
    name_pandas_file = "DBSCAN_new_metric_position_and_velocity_table_results"
    column_names = ["alpha", "phi", "mean ARI score"]

    # create empty dataframe
    results = pd.DataFrame(columns=column_names)

    global phi
    global alpha

    eps = 85
    min_sample = 2
    for phi_ in list_phi:
        for alpha_ in list_alpha:

            phi = phi_
            alpha = alpha_

            # produce results for DBSCAN algorithm with these values of eps and min_sample
            old_labels = DBscan_step_intuition_dist(steps[0], None, directory,
                                                    min_sample=min_sample,
                                                    eps=eps)

            for step in steps[1:]:
                old_labels = DBscan_step_intuition_dist(step,
                                                        old_labels,
                                                        directory,
                                                        min_sample=min_sample,
                                                        eps=eps)

            # produce results
            filename_true = "ground_truth_label"
            filename_pred = "DBSCAN_intuition_dist_phi=" + str(phi) + "_alpha=" + str(alpha) + "_label"

            score = calculate_rand_score(steps, directory, filename_true, filename_pred)
            results = results.append({"alpha": alpha, "phi": phi,
                                      "mean ARI score": score}, ignore_index=True)
            print(results)
    # stock dataframe into a file
    results.to_csv(name_pandas_file + ".csv", index=False)


def build_ground_truth(step, old_labels, repository, list_nb_boids,
                       beta=23,
                       eps=75,
                       min_sample=2):
    """
    build ground truth with DBscan on positions
    """
    positions, velocities, headings = \
        get_positions_velocity_headings(repository, step)

    labels = np.zeros(positions.shape[0], dtype=int)

    sum_boids = 0

    for nb_boids in list_nb_boids:

        indices = np.arange(sum_boids, sum_boids + nb_boids)

        sum_boids = sum_boids + nb_boids
        """
        train_data = np.concatenate((positions[indices],
                                     beta * velocities[indices]),
                                    axis=1)
        """
        train_data = positions[indices]

        db = DBSCAN(eps=eps, min_samples=min_sample).fit(train_data)

        for ind_0, ind in zip(np.arange(0, nb_boids), indices):
            #  we keep the zeros, we apply + 50 to the other labels
            labels[ind] = np.where(db.labels_[ind_0] > -1,
                                   db.labels_[ind_0] + sum_boids, 0)
            # for getting rid of -1 labels
            # we apply db.labels + sum_boids to differentiate clusters
            # from different species

        if old_labels is not None:
            labels[indices] = merge_labels(old_labels[indices],
                                           labels[indices])

    stock_labels(labels, step, repository=repository,
                 filename="ground_truth_label")

    return labels


def calculate_rand_score(steps, repository, filename_true, filename_pred):
    """
    Compute Rand index at each step specified
    steps: list of step to compute
    repository: repository where the labels are
    filename_1: name of files for first labels
    filename_2: name of files for second labels
    output_file: file with Rand index scores for each steps
    """
    # for all files
    list_scores = list()
    for step in steps:
        labels_true = charge_labels_simulation(repository, filename_true, step)
        labels_pred = charge_labels_simulation(repository, filename_pred, step)

        score = adjusted_rand_score(labels_true, labels_pred)
        list_scores.append(score)

    scores = np.array(list_scores)
    # save scores into a file
    filename = "scores/" + "_ARIscore_" + filename_pred + filename_true
    print("mean ARI score:", np.mean(scores))
    np.savetxt(filename, scores)
    return np.mean(scores)


if __name__ == "__main__":
    # steps to test the clustering algorithm
    steps = list(np.arange(300, 1000))

    # directory where the results will be stored.
    directory = "simulation_data_200_Boids/"
    """
    list_eps = [65, 70, 75, 80, 85, 90]
    list_min_sample = [2, 3, 4, 5]

    # run test for DBSCAN using positions
    test_DBSCAN_positions(directory=directory,
                          steps=steps,
                          list_eps=list_eps,
                          list_min_sample=list_min_sample)

    list_alpha = [0.8, 1, 1.2]
    list_beta = [10, 15, 20, 25, 30, 35, 40]
    list_eps = [85]
    list_min_sample = [2]

    # run test for DBSCAN on positions and velocities
    test_DBSCAN_positions_and_velocity(directory=directory,
                                       steps=steps,
                                       list_alpha=list_alpha,
                                       list_beta=list_beta,
                                       list_eps=list_eps,
                                       list_min_sample=list_min_sample)
    """
    list_phi = [50, 100, 150, 200]
    list_alpha = [0.8, 1, 1.2]

    test_DBSCAN_new_metric_positions_and_velocity(directory=directory,
                                                  steps=steps,
                                                  list_alpha=list_alpha,
                                                  list_phi=list_phi)
