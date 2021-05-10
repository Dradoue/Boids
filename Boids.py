import numpy as np

from Physics2D import Physics2D
from constants import WINDOW_SIZE, \
    BOID_VIEW_ANGLE, BOID_VIEW, \
    EPSILON, LEAF_SIZE, \
    SEPARATION_DIST, \
    SEPARATION_FORCE, \
    COHESION_FORCE, ALIGNMENT_FORCE, \
    CHASING_FORCE, FLEE_FORCE, \
    RELATIONS_SPECIES
# imports from local files
from utils import angle_between


class Boids(Physics2D):
    """
    Boids class, inherited from Physics2D, define the behavior of boids

    :param num_entities: number of boids for each species, (ex: [12,13] mean 12 boids for species 1 and 
    13 for species 2)
    :param positions: initial positions of boids, 2D array of shape (n_boids, 2)
    :param velocities: initial velocities of boids, 2D array of shape (n_boids, 2)
    :param min_speed: min speed (l2-norm) of boids
    :param max_speed: max speed (l2-norm) of boids
    :param max_turn: max angle change a boid can have on one step, pi/48 by default.
    """
    def __init__(self, num_entities, positions, velocities,
                 min_speed, max_speed, max_turn, dt=1):

        super().__init__(positions, velocities,
                         min_speed, max_speed, max_turn, dt)

        self.positions = positions

        self.list_number_boids = num_entities

        self.grid = \
            [[[] for _ in range(int((WINDOW_SIZE[0] / LEAF_SIZE)))]
             for _ in range(int(WINDOW_SIZE[1] / LEAF_SIZE))]

        self.steering = np.zeros(shape=(self.nb_entities, 2))

    def update_boids(self):
        """
        calculate acceleration for each boid with the rules.
        """
        self.update_grid()
        # calculate steering with rules

        # apply sca for each type of boids
        self.apply_sca()
        self.update(self.steering)

    def apply_sca(self):
        """
        apply separation, cohesion, alignment, chase and flee rules
        """
        self.steering = np.zeros(shape=(self.nb_entities, 2))

        for i in range(int(WINDOW_SIZE[0] / LEAF_SIZE)):
            for j in range(int(WINDOW_SIZE[1] / LEAF_SIZE)):

                sum = 0
                grid = self.grid[i][j]
                list_species_indices = list()
                # for each species we apply CSA.
                n_species = 0
                for num_boids in self.list_number_boids:

                    boids_indices = np.arange(sum, sum + num_boids)
                    sum += num_boids

                    list_ind = list()
                    for ind in boids_indices:
                        if ind in grid:
                            list_ind.append(ind)

                    list_species_indices.append(list_ind)

                    self.apply_cohesion(list_ind, n_species)
                    self.apply_alignment(list_ind, n_species)
                    self.apply_separation(list_ind, n_species)

                    n_species += 1

                # apply chasing and fleeing relations
                for i in range(RELATIONS_SPECIES.shape[0]):
                    for j in range(RELATIONS_SPECIES.shape[0]):

                        # species i chase species j
                        if RELATIONS_SPECIES[i, j] == 1:

                            self.apply_chasing(list_species_indices[i], list_species_indices[j], i)

                        # species i flee species j
                        if RELATIONS_SPECIES[i, j] == -1:

                            self.apply_fleeing(list_species_indices[i], list_species_indices[j], i)

    def update_grid(self):
        """
        update the grid
        """
        self.grid = [[[] for _ in range(int((WINDOW_SIZE[0] / LEAF_SIZE)))]
                     for _ in range(int(WINDOW_SIZE[1] / LEAF_SIZE))]

        indices_ = np.concatenate((np.array(np.floor(self.positions[:, 0] / LEAF_SIZE), dtype=int)[:, np.newaxis],
                                   np.array(np.floor(self.positions[:, 1] / LEAF_SIZE), dtype=int)[:, np.newaxis],
                                   np.array(range(self.nb_entities), dtype=int)[:, np.newaxis]), axis=1)

        for i, j, k in indices_:
            self.grid[i][j].append(k)

    def apply_cohesion(self, indices, n_specie):
        """
        apply cohesion rule to boids which have indices :indices: for species :n_specie:
        :param indices: indices to apply the rule
        :n_specie: the species, an integer between 0 and num_species (used for parameters)
        """
        if len(indices) <= 1:
            return

        for ind in indices:

            indices_ = list(indices)
            indices_.remove(ind)
            indices_ = np.array(indices_)

            dist = np.linalg.norm(self.positions[indices_, :]
                                  - self.positions[ind, :], axis=1)

            # neighbors indices relatively to indices_
            neighbors = np.where(dist < BOID_VIEW[n_specie])[0]

            if neighbors.shape[0] > 0:

                # true_neighbors: true indices for positions and velocities
                # neighbors: indices of neighbors relatively to indices_
                true_neighbors = indices_[neighbors]

                diff = self.positions[true_neighbors, :] \
                       - self.positions[ind, :]

                with np.errstate(invalid='ignore'):
                    respect_angles = \
                        np.where(angle_between(self.velocities[ind, :], diff)
                                 <= BOID_VIEW_ANGLE[n_specie] / 2)[0]

                # respect_angles: indices of neighbors
                # relatively to true_neighbors
                final_true_neighbors = true_neighbors[respect_angles]

                if final_true_neighbors.shape[0] > 0:

                    cohesion = np.mean(self.positions[final_true_neighbors, :],
                                       axis=0) - self.positions[ind, :]

                    norm_cohesion = np.linalg.norm(cohesion)

                    if norm_cohesion > EPSILON:
                        self.steering[ind, :] += COHESION_FORCE[n_specie] * \
                                                 (cohesion / norm_cohesion)

    def apply_separation(self, indices, n_specie):
        """
        apply separation rule to boids which have indices :indices: for species number :n_specie:
        :param indices: indices of boids to apply the rule
        :n_specie: the species, an integer between 0 and num_species (used for parameters)
        """
        if len(indices) <= 1:
            return
        for ind in indices:

            indices_ = list(indices)
            indices_.remove(ind)
            indices_ = np.array(indices_)

            # get the distances
            # l2 norms.
            dist_norm = np.linalg.norm(self.positions[indices_, :]
                                       - self.positions[ind, :], axis=1)
            dist = self.positions[indices_, :] \
                   - self.positions[ind, :]

            neighbors = np.where(dist_norm <= SEPARATION_DIST[n_specie])[0]

            if neighbors.shape[-1] != 0:

                respect_angles = \
                    np.where(angle_between(self.velocities[ind, :], dist[neighbors])
                             <= BOID_VIEW_ANGLE[n_specie] / 2)[0]

                #  neighbors that respect angles respectively to indices_
                neighbors_that_respect_angle = neighbors[respect_angles]

                if neighbors_that_respect_angle.shape[0] > 0:
                    sep = np.sum(dist[neighbors_that_respect_angle, :], axis=0)

                    self.steering[ind, :] -= SEPARATION_FORCE[n_specie] * sep
                    # sep/dist_norm?

    def apply_alignment(self, indices, n_specie):
        """
        apply alignment rule to boids which have indices :indices: for species number :n_specie:
        alignment -> boids tends to move like their neighbors
        :param indices: indices of boids to apply the rule
        :n_specie: the species, an integer between 0 and num_species (used for parameters)
        """
        if len(indices) <= 1:
            return

        for ind in indices:

            indices_ = list(indices)
            indices_.remove(ind)
            indices_ = np.array(indices_)

            dist = np.linalg.norm(self.positions[indices_, :]
                                  - self.positions[ind, :], axis=1)
            neighbors = np.where(dist < BOID_VIEW[n_specie])[0]
            #  neighbors relatively to indices_

            if neighbors.shape[-1] != 0:

                true_indices_neighbors = indices_[neighbors]

                diff = self.positions[true_indices_neighbors, :] \
                       - self.positions[ind, :]

                respect_angles = \
                    np.where(angle_between(self.velocities[ind, :], diff)
                             <= BOID_VIEW_ANGLE[n_specie] / 2)[0]

                #  neighbors relatively to true_indices_neighbors
                final_true_neighbors = true_indices_neighbors[respect_angles]

                if final_true_neighbors.shape[0] > 0:

                    mean_vel = np.mean(self.velocities[final_true_neighbors, :], axis=0)
                    norm_vel = np.linalg.norm(mean_vel)

                    if norm_vel > EPSILON:
                        self.steering[ind, :] += ALIGNMENT_FORCE[n_specie] * \
                                                 (mean_vel / norm_vel)

    def apply_chasing(self, indices_chase, indices_chased, n_specie):
        """
        apply alignment rule to boids which have indices :indices: for species number :n_specie:
        alignment -> boids tends to move like their neighbors
        :param indices: indices of boids to apply the rule
        :n_specie: the species, an integer between 0 and num_species (used for parameters)
        """
        if len(indices_chase) == 0 or len(indices_chased) == 0:
            return

        # we take each that are chasing
        for ind in indices_chase:

            indices_ = list(indices_chased)
            indices_ = np.array(indices_)

            dist = np.linalg.norm(self.positions[indices_, :]
                                  - self.positions[ind, :], axis=1)

            # neighbors indices relatively to indices_
            neighbors = np.where(dist < BOID_VIEW[n_specie])[0]

            if neighbors.shape[0] > 0:

                # true_neighbors: true indices for positions and velocities
                # neighbors: indices of neighbors relatively to indices_
                true_neighbors = indices_[neighbors]

                diff = self.positions[true_neighbors, :] \
                       - self.positions[ind, :]

                with np.errstate(invalid='ignore'):
                    respect_angles = \
                        np.where(angle_between(self.velocities[ind, :], diff)
                                 <= BOID_VIEW_ANGLE[n_specie] / 2)[0]

                # respect_angles: indices of neighbors
                # relatively to true_neighbors
                final_true_neighbors = true_neighbors[respect_angles]

                if final_true_neighbors.shape[0] > 0:

                    cohesion = np.mean(self.positions[final_true_neighbors, :],
                                       axis=0) - self.positions[ind, :]

                    norm_cohesion = np.linalg.norm(cohesion)

                    if norm_cohesion > EPSILON:
                        self.steering[ind, :] += CHASING_FORCE[n_specie] * \
                                                 (cohesion / norm_cohesion)

    def apply_fleeing(self, indices_flee, indices_fleed, n_specie):

        if len(indices_flee) == 0 or len(indices_fleed) == 0:
            return

        # we take each that are fleeing
        for ind in indices_flee:

            indices_ = list(indices_fleed)
            indices_ = np.array(indices_)

            # get the distances
            dist_norm = np.linalg.norm(self.positions[indices_, :]
                                       - self.positions[ind, :], axis=1)

            dist = self.positions[indices_, :] \
                   - self.positions[ind, :]

            neighbors = np.where(dist_norm <= BOID_VIEW[n_specie])[0]

            if neighbors.shape[-1] != 0:

                respect_angles = \
                    np.where(angle_between(self.velocities[ind, :], dist[neighbors])
                             <= BOID_VIEW_ANGLE[n_specie] / 2)[0]

                #  neighbors that respect angles respectively to indices_
                neighbors_that_respect_angle = neighbors[respect_angles]

                if neighbors_that_respect_angle.shape[0] > 0:
                    flee = np.mean(dist[neighbors_that_respect_angle, :], axis=0)

                    norm_flee = np.linalg.norm(flee)

                    self.steering[ind, :] -= FLEE_FORCE[n_specie] * flee / norm_flee
