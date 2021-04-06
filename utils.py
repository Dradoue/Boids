import math

import numpy as np
import os
from constants import *


def toroid_dist_between_points(a, b):
    pos = [0, 0]
    if a[0] > b[0]:
        pos[0] = np.minimum(b[0] + (WINDOW_SIZE[0] - a[0]), a[0] - b[0])
    else:
        pos[0] = np.minimum(a[0] + (WINDOW_SIZE[0] - b[0]), b[0] - a[0])

    if a[1] > b[1]:
        pos[1] = np.minimum(b[1] + (WINDOW_SIZE[1] - a[1]), a[1] - b[1])
    else:
        pos[1] = np.minimum(a[1] + (WINDOW_SIZE[1] - b[1]), b[1] - a[1])
    return np.linalg.norm(pos)


def is_empty(repository):
    files = os.listdir(repository)
    return len(files) == 0


def erase_files(repository):
    files = os.listdir(repository)
    for file in files:
        os.remove(repository + '/' + file)


def angle_between(a, b):
    """
    a: vector
    b: vector
    return relative angles between a and b
    """
    # calculate angles vector between (x, y) coordinates of a and b
    denom = np.linalg.norm(a) * np.linalg.norm(b, axis=1)[None, :].T

    a = np.array([a] * denom.shape[0]).reshape((denom.shape[0], 2))

    dot_product = np.sum(a * b, axis=1)

    res = dot_product[None, :].T / denom
    with np.errstate(invalid='ignore'):
        res = np.arccos(res)

    res = np.degrees(res)

    return res


def cart2pol(vector):
    """
    Cartesian to polar coordinates for a vector
    vector: 2D array with x, y coordinates
    rho: norm of the coordinates
    phi: angle of the coordinates
    """
    rho = np.sum(vector ** 2, axis=-1) ** (1. / 2)

    phi = np.arctan2(vector[:, 1], vector[:, 0])

    return rho, phi


def pol2cart(rho, phi):
    """
    Polar to cartesian coordinates with vector
    rho: vector of norm, float
    phi: vector of angles, rad
    """
    x = np.array(rho * np.cos(phi))[:, np.newaxis]
    y = np.array(rho * np.sin(phi))[:, np.newaxis]

    return np.concatenate((x, y), axis=1)


def adjust_position_to_boundaries(positions, bounds, tolerance=DEFAULT_TOLERANCE):
    """
    Function to update boid position if crossing a boundary (toroid boundary condition)
    :param positions: vector of (x,y) positions
    :param bounds: (xmin,xmax,ymin,ymax) boundaries
    :param tolerance: optional tolerance for being on boundary. by default set to DEFAULT_TOLERANCE (in constants.py)
    """
    positions[:, 0] = np.where(positions[:, 0] < (bounds[0] - tolerance), positions[:, 0] + bounds[1])[0]

    positions[:, 0] = np.where(positions[:, 0] > (bounds[1] - tolerance), positions[:, 0] - bounds[1])[0]

    positions[:, 1] = np.where(positions[:, 1] < (bounds[2] - tolerance), positions[:, 1] + bounds[3])[0]

    positions[:, 1] = np.where(positions[:, 1] > (bounds[3] + tolerance), positions[:, 1] - bounds[3])[0]

    return positions


def generate_aleatory_positions(num_pos):
    """
    Generate aleatory positions within the window
    num_pos: number of positions to generate
    """
    return [(np.random.uniform(20, WINDOW_SIZE[0] - 20), np.random.uniform(20, WINDOW_SIZE[1] - 20))
            for _ in range(num_pos)]


def generate_aleatory_angles(num_headings):
    """
    Generate a list of aleatory headings in rad
    num_headings: number of heading to generate
    """
    return [math.pi * (np.random.randint(1, 360) / 180) for _ in range(num_headings)]


def generate_list_aleatory_points(num_steering, min_x, min_y, max_x, max_y):
    """
    Generate a vector filled with num_steering (x, y) points between min_x, min_y and max_x, max_y
    """
    return [(np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)) for _ in range(num_steering)]


def get_positions_velocity_headings(repository, filename, step):
    try:
        with open("data/" + repository + filename + "positions_" + str(step)):

            headings = np.array(np.loadtxt("data/" + repository + filename
                                           + "headings_" + str(step)), dtype=float)
            positions = np.array(
                np.loadtxt("data/" + repository + filename + "positions_" + str(step)), dtype=float).reshape(
                (headings.shape[0], 2))
            velocities = np.array(
                np.loadtxt("data/" + repository + filename + "velocities_" + str(step)), dtype=float).reshape(
                (headings.shape[0], 2))

            return positions, velocities, headings

    except IOError:
        print("Could not open file for step {0}, simulation terminated".format(str(step)))
        exit()


def charge_labels_simulation(repository, file_label_name, step):
    """
    used to charge label files
    """
    try:
        with open(repository + file_label_name + str(step)):

            labels = np.array(np.loadtxt("data/" + repository + file_label_name
                                         + str(step)), dtype=int)

            return labels

    except IOError:
        print("Could not open file named {0}".format(file_label_name))
        exit()


def get_cluster_partitions(repository, file_label_name, step):
    """
    return list of partitions (which are list of indices)
    """
    labels = charge_labels_simulation(repository, file_label_name, step)

    res = list()

    for ind in np.sort(np.unique(labels)):
        indices = list(np.where(labels == ind)[0])
        res.append(indices)

    return res


def write_constants_into_simulation_directory(directory):

    f = open("constants.py", "r")
    text = f.read()
    f.close()
    f = open("data/" + directory + "constants_used_for_this_simulation.txt", "w+")
    f.write(text)
    f.close()
