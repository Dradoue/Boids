import collections
import karateclub
import networkx as nx
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from st_dbscan import ST_DBSCAN
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

import keras
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv1D

from constants import TRIANGLES_COLORS
from utils import toroid_dist_between_points, \
    get_positions_velocity_headings, charge_labels_simulation

global phi
global alpha
phi = 100
alpha = 1.2


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


def calculate_edges_1(positions, velocities, headings):
    """
    simple method for calculating edges with positions,
    velocities and headings to be used later
    """
    list_edges = []
    for i in range(positions.shape[0]):
        for j in list(range(positions.shape[0])):
            if i != j:
                if toroid_dist_between_points(positions[i, :],
                                              positions[j, :]) < 150:
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


def graph_step(step, repository="simulation_data/", filename="file"):
    positions, velocities, headings = \
        get_positions_velocity_headings(repository, filename, step)

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

def merging_labels(old_labels, new_labels):
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


def stock_file(labels, step, repository, filename):
    np.savetxt(repository + filename + str(step), labels)


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
def d1(a1, a2):
    res = np.linalg.norm(a1[:2] - a2[:2])
    return res


def d2(a1, a2):
    similarity = np.dot(a1[2:], a2[2:]) / (np.linalg.norm(a1[2:]) * np.linalg.norm(a2[2:]))
    angular_dist = np.abs(similarity - 1)
    return angular_dist


def linear_comb_dist12(a1, a2):
    global phi, alpha
    return alpha * d1(a1, a2) + phi * d2(a1, a2)


def linear_comb_dist12_multiplestep(a1, a2, nb_step=3, gamma=0.5):
    # a1 and a2 are matrices with multiple timesteps data
    res = 0
    for i in range(0, nb_step):
        res = res + gamma ** (nb_step - i) * linear_comb_dist12(a1[i:i + 2], a2[i:i + 2])
    return res


def DBscan_step_positions(step, old_labels, repository, filename):
    """
    DBscan algorithm on positions
    """
    positions, velocities, headings = \
        get_positions_velocity_headings(repository, filename, step)
    # train_data = np.concatenate((positions, velocities), axis=1)
    train_data = positions
    db = DBSCAN(eps=85, min_samples=2).fit(train_data)
    labels = db.labels_ + 1  # for getting rid of -1 labels
    if old_labels is not None:
        labels = merging_labels(old_labels, labels)
    stock_file(labels, step, repository=repository,
               filename="DBSCAN_positions_label")

    return labels


def DBscan_step_positions_and_velocity(step, old_labels, repository,
                                       filename, beta=27):
    """
    DBcsan algorithm on positions + beta * velocities
    """
    positions, velocities, headings = \
        get_positions_velocity_headings(repository, filename, step)

    train_data = np.concatenate((positions, beta * velocities), axis=1)

    db = DBSCAN(eps=85, min_samples=2).fit(train_data)
    labels = db.labels_ + 1  # for getting rid of -1 labels
    if old_labels is not None:
        labels = merging_labels(old_labels, labels)
    stock_file(labels, step, repository=repository,
               filename="DBSCAN_position|velocity_label")

    return labels


def ST_DBSCAN_step(step, old_labels, repository, filename):
    """
    perform st-DBSCAN on data
    """
    positions, velocities, headings = \
        get_positions_velocity_headings(repository, filename, step)

    train_data = positions
    st_db = ST_DBSCAN(eps1=85, eps2=85, min_samples=2).fit(train_data)
    labels = st_db.labels_ + 1  # for getting rid of -1 labels
    if old_labels is not None:
        labels = merging_labels(old_labels, labels)
    stock_file(labels, step, repository=repository,
               filename="ST-DBSCAN_position_label")

    return labels


def DBscan_step_intuition_dist(step, old_labels, repository,
                               filename, min_sample=2, eps=85):
    """
    DBcsan algorithm on positions + beta * velocities
    """
    global phi, alpha
    positions, velocities, headings = \
        get_positions_velocity_headings(repository, filename, step)

    train_data = np.concatenate((positions, velocities), axis=1)

    db = DBSCAN(eps=eps, min_samples=min_sample, metric=linear_comb_dist12).fit(train_data)

    labels = db.labels_ + 1  # for getting rid of -1 labels
    if old_labels is not None:
        labels = merging_labels(old_labels, labels)
    stock_file(labels, step, repository=repository,
               filename="DBSCAN_intuition_dist_phi=" + str(phi) + "_alpha=" + str(alpha) + "_label")

    return labels


def DBscan_step_intuition_dist_multistep(step, old_labels, repository,
                                         filename, min_sample=2, eps=85, nb_step=3):
    """
    DBcsan algorithm on positions + beta * velocities
    """
    global phi, alpha

    train_data = None

    for i in range(step, step + nb_step, 1):

        positions, velocities, headings = \
            get_positions_velocity_headings(repository, filename, step)

        if train_data is not None:
            train_data = np.concatenate((train_data, np.concatenate((positions, velocities), axis=1)),
                                        axis=0)
        else:
            train_data = np.concatenate((positions, velocities), axis=1)
    db = DBSCAN(eps=eps, min_samples=min_sample, metric=linear_comb_dist12_multiplestep).fit(train_data)

    labels = db.labels_ + 1  # for getting rid of -1 labels
    if old_labels is not None:
        labels = merging_labels(old_labels, labels)
    stock_file(labels, step, repository=repository,
               filename="DBSCAN_intuition_dist_phi=" + str(phi) + "_alpha=" + str(alpha) + "_label")

    return labels


def DBscan_step_intuition_dist(step, old_labels, repository,
                               filename, min_sample=2, eps=85):
    """
    DBcsan algorithm on positions + beta * velocities
    """
    global phi, alpha
    positions, velocities, headings = \
        get_positions_velocity_headings(repository, filename, step)

    train_data = np.concatenate((positions, velocities), axis=1)

    db = DBSCAN(eps=eps, min_samples=min_sample, metric=linear_comb_dist12).fit(train_data)

    labels = db.labels_ + 1  # for getting rid of -1 labels
    if old_labels is not None:
        labels = merging_labels(old_labels, labels)
    stock_file(labels, step, repository=repository,
               filename="DBSCAN_intuition_dist_phi=" + str(phi) + "_alpha=" + str(alpha) + "_label")

    return labels


def build_ground_truth(step, old_labels, repository,
                       filename, list_nb_boids, beta=23):
    """
    build ground truth with DBscan on positions
    """
    positions, velocities, headings = \
        get_positions_velocity_headings(repository, filename, step)

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

        db = DBSCAN(eps=85, min_samples=2).fit(train_data)

        for ind_0, ind in zip(np.arange(0, nb_boids), indices):
            #  we keep the zeros, we apply + 50 to the other labels
            labels[ind] = np.where(db.labels_[ind_0] > -1,
                                   db.labels_[ind_0] + sum_boids, 0)
            # for getting rid of -1 labels
            # we apply db.labels + sum_boids to differentiate clusters
            # from different species

        if old_labels is not None:
            labels[indices] = merging_labels(old_labels[indices],
                                             labels[indices])

    stock_file(labels, step, repository=repository,
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
    filename = "_ARIscore_" + filename_pred + filename_true
    print("mean ARI score:", np.mean(scores))
    np.savetxt(filename, scores)


from keras.layers import Input, Conv1D, Dropout, Conv1DTranspose, Dense, MaxPooling1D, Flatten, BatchNormalization
from keras.models import Model


def create_dataset(repository="simulation_data_30-30-30-30/", filename="",
                   step_init=300, step_end=6500):
    # build labels
    labels = np.array([0] + [1] + [2] + [3])[:, np.newaxis]
    enc = OneHotEncoder()
    enc.fit(X=labels)
    labels_encoded = enc.transform(labels).toarray()

    use_headings = True

    # build dataset of T * n * (positions + headings)
    data = list()
    labels = list()

    for t in range(step_init, step_end, 1):
        positions, velocities, headings = \
            get_positions_velocity_headings(repository, filename, t)

        # normalise data
        positions = positions / 2000
        headings = headings / np.pi

        # add new axis to headings for dimension match
        headings_1 = headings[0:30, np.newaxis]
        headings_2 = headings[30:60, np.newaxis]
        headings_3 = headings[60:90, np.newaxis]
        headings_4 = headings[90:120, np.newaxis]

        positions_1 = positions[0:30, :]
        positions_2 = positions[30:60, :]
        positions_3 = positions[60:90, :]
        positions_4 = positions[90:120, :]

        if use_headings:
            # concatenate positions and headings
            data_1 = np.concatenate((positions_1, headings_1), axis=1).reshape(1, 90)
            data_2 = np.concatenate((positions_2, headings_2), axis=1).reshape(1, 90)
            data_3 = np.concatenate((positions_3, headings_3), axis=1).reshape(1, 90)
            data_4 = np.concatenate((positions_4, headings_4), axis=1).reshape(1, 90)

            data.append(data_1)
            data.append(data_2)
            data.append(data_3)
            data.append(data_4)

            labels.append(labels_encoded[0, :])
            labels.append(labels_encoded[1, :])
            labels.append(labels_encoded[2, :])
            labels.append(labels_encoded[3, :])

        else:

            data.append(positions_1.reshape(1, 60))
            data.append(positions_1.reshape(1, 60))
            data.append(positions_1.reshape(1, 60))
            data.append(positions_1.reshape(1, 60))

            labels.append(labels_encoded[0, :])
            labels.append(labels_encoded[1, :])
            labels.append(labels_encoded[2, :])
            labels.append(labels_encoded[3, :])

    data_ = np.array(data, dtype=float)
    labels_ = np.array(labels, dtype=int)
    print("data and labels built")
    print("data shape ", data_.shape)
    print("labels shape ", labels_.shape)
    print(data_)
    print(labels_)

    if use_headings:
        np.savetxt("data.txt", data_.reshape(((6500 - 300) * 4, 90)))
        np.savetxt("targets.txt", labels_.reshape(((6500 - 300) * 4, 4)))
    else:
        np.savetxt("data.txt", data_.reshape(((6500 - 300) * 4, 60)))
        np.savetxt("targets.txt", labels_.reshape(((6500 - 300) * 4, 4)))

    return data_, labels_


def build_network_dense():
    inputs = keras.Input(shape=(90,))
    x = Dense(5, activation="relu")(inputs)
    x = Dense(5, activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(4, activation="softmax", name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    model.summary()
    return model


def build_conv1D_network():
    model = Sequential()
    model.add(Conv1D(4, 25, activation="relu", input_shape=(90, 1)))
    model.add(BatchNormalization())
    model.add(Conv1D(4, 25, activation="relu"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    model.summary()
    return model


from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd


def train_network(model, data, targets):
    choice_split = 1

    print(data)
    print(data.shape)

    X_train, X_test, y_train, y_test = None, None, None, None

    if choice_split == 0:

        X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.4, random_state=1)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.4, random_state=1)

    else:

        X_train, X_test, X_val, y_train, y_test, y_val = data[:15000, :], data[15000:20000, :], data[20000:, :], \
                                                         targets[:15000, :], targets[15000:20000, :], targets[20000:, :]

    history = model.fit(
        X_train,
        y_train,
        batch_size=64,
        epochs=50,
        verbose=1,
        shuffle=True,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(X_val, y_val),
    )

    score = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict_classes(X_test)

    confmat = confusion_matrix(y_pred, np.argmax(y_test, axis=1))

    df_cm = pd.DataFrame(confmat)
    print(confmat)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save_weights('model.h5')

    return model


def experiment_network():
    data, targets = create_dataset(repository="simulation_data_30-30-30-30/")
    # data = np.loadtxt("data.txt", dtype=float)
    # targets = np.loadtxt("targets.txt")
    data = data.reshape((24800, 90, 1))
    model = build_conv1D_network()
    train_network(model, data, targets)


if __name__ == '__main__':
    experiment_network()
