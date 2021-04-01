import numpy as np

from constants import BOUNDS, DEFAULT_TOLERANCE
from utils import cart2pol, pol2cart


class Physics2D:
    """
    Physics2D class

    Physical model used for realism of boids trajectories

    :param

    positions: numpy array of (x, y) positions
    velocity: numpy array of (vx, vy) velocities
    min_speed: minimum speeds allowed
    max_speed: max speeds allowed
    num_entities: list of len for number of each type (color) of boids
    """

    def __init__(self, positions, velocities, min_speed, max_speed, max_turn, dt=1):
        self.nb_entities = positions.shape[0]
        # set limits of speed
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.dt = dt

        # set max turn, rad
        self.max_turn = max_turn

        # set position, acceleration and velocity
        self.positions = np.array(positions)
        self.accelerations = np.zeros(self.positions.shape)

        #  euclidean velocities
        self.velocities = np.array(velocities)

        #  angles of boids, rad
        self.headings = None

    def update(self, steering):
        """
        Update the positions and headings
        steering: vector of (x, y) steering
        """
        self.accelerations = steering * self.dt

        # old heading
        _, old_headings = cart2pol(self.velocities)

        new_velocities = self.velocities + self.accelerations * self.dt

        # new velocity in polar coordinates
        speeds, new_headings = cart2pol(new_velocities)

        # calculate heading diff
        headings_diff = np.radians(180 - (180 - np.degrees(new_headings)
                                          + np.degrees(old_headings)) % 360)

        # if the heading diff is too big, we cut the turn at max_turn or min_turn
        # use np.where
        new_headings = np.where(headings_diff < -self.max_turn,
                                old_headings - self.max_turn, new_headings)

        new_headings = np.where(self.max_turn < headings_diff,
                                old_headings + self.max_turn, new_headings)

        # if the speed is too slow or too big, we adjust it
        # use np.where
        speeds = np.where(speeds < self.min_speed, self.min_speed, speeds)
        speeds = np.where(speeds > self.max_speed, self.max_speed, speeds)

        # set the new velocity
        # modify pol2cart
        self.velocities = pol2cart(speeds, new_headings)

        self.headings = new_headings
        # move
        self.positions += self.velocities * self.dt

        self.adjust_positions_to_boundaries()

    def adjust_positions_to_boundaries(self):
        """
        Method for boundary conditions
        (toroid boundary condition)
        """
        self.positions[:, 0] = \
            np.where(self.positions[:, 0] < BOUNDS[0]
                     + DEFAULT_TOLERANCE, self.positions[:, 0]
                     + BOUNDS[1], self.positions[:, 0])

        self.positions[:, 0] = \
            np.where(self.positions[:, 0] > BOUNDS[1]
                     - DEFAULT_TOLERANCE, self.positions[:, 0]
                     - BOUNDS[1], self.positions[:, 0])

        self.positions[:, 1] = \
            np.where(self.positions[:, 1] < BOUNDS[2]
                     + DEFAULT_TOLERANCE, self.positions[:, 1] +
                     BOUNDS[3], self.positions[:, 1])

        self.positions[:, 1] = \
            np.where(self.positions[:, 1] > BOUNDS[3]
                     - DEFAULT_TOLERANCE, self.positions[:, 1] -
                     BOUNDS[3], self.positions[:, 1])
