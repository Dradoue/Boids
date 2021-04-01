# -----------------------------------------------------------
# email: edouard.gouteux@gmail.com
# -----------------------------------------------------------

import math

import numpy as np
import pyglet
import pyglet.gl as GL

from Boids import Boids
from Triangles import Triangles
from constants import WINDOW_SIZE, \
    BOUNDS
# imports from local files
from utils import get_positions_velocity_headings, \
    is_empty, EraseFile
from ML import graph_step, DBscan_step_positions, \
    DBscan_step_positions_and_velocity, \
    labels_to_colorlist, build_ground_truth, \
    DBscan_step_intuition_dist, DBscan_step_intuition_dist_multistep


class Simulation:
    """
    class Simulation: create a simulation, stock the data, apply some ML, play scenarios

    :param list_num_boids: (optional) list containing the number of boids for each species.
    :param min_speed: (optional) min speed for all boids, 0.2 by default.
    :param max_speed: (optional) max speed for all boids, 4 by default.
    :param max_turn: (optional) max angle change a boid can have on one step, pi/48 by default.
    :param repository: (optional) repository where we will stock the simulation data (=positions, headings, velocities)
    :param positions_init: (optional) initial positions of boids
    :param velocities_init: (optional) initial velocities of boids
    :param step: (optional) initial step of the simulation, 1 by default
    """

    def __init__(self, **kwargs):
        """
        initialise all parameters
        """
        self.list_num_boids = kwargs.get('list_num_boids',
                                         [150, 150])

        self.number_boids = np.sum(self.list_num_boids)

        self.min_speed = kwargs.get('min_speed', .2)
        self.max_speed = kwargs.get('max_speed', 4)
        self.max_turn = kwargs.get('max_turn', math.pi / 48)
        self.filename = kwargs.get('file_name', "") # to dégager.
        self.repository = kwargs.get('repository',
                                     "simulation_data/")

        self.path = self.repository + self.filename

        self.positions_init = kwargs.get('positions_init',
                                         np.array([(np.random.uniform(BOUNDS[0],
                                                                      BOUNDS[1]),
                                                    np.random.uniform(BOUNDS[2],
                                                                      BOUNDS[3]))
                                                   for _ in range(self.number_boids)]))

        self.velocities_init = kwargs.get('velocities_init',
                                          np.array([(np.random.uniform(-self.max_speed,
                                                                       self.max_speed),
                                                     np.random.uniform(-self.max_speed,
                                                                       self.max_speed))
                                                    for _ in range(self.number_boids)]))

        self.boids = Boids(self.list_num_boids,
                           self.positions_init, self.velocities_init,
                           self.min_speed,
                           self.max_speed,
                           max_turn=self.max_turn)

        self.triangles = self.triangles = \
            Triangles(list_num_triangles=self.list_num_boids)
        # Set background to white

        self.step = kwargs.get('step', 1)  # step to begin the simulation

        self.old_labels = None  # a help for clusters labels

    def stock_positions_velocities_headings_and_increase_step(self):
        """
        In the title
        """
        np.savetxt(self.path + "positions_" + str(self.step),
                   self.boids.positions, fmt="%f")
        np.savetxt(self.path + "velocities_" + str(self.step),
                   self.boids.velocities, fmt="%f")
        np.savetxt(self.path + "headings_" + str(self.step),
                   self.boids.headings, fmt="%f")
        self.step += 1

    #################################################################
    # some animation functions for different scenarios
    #################################################################

    def animate(self, time):
        """
        Standard animate step: we update boids,
        stock standard data (positions, velocity, headings) in self.repository,
        with prefix self.filename on the files, we increment steps value, and update the screen
        """
        self.boids.update_boids()
        self.stock_positions_velocities_headings_and_increase_step()
        self.triangles.update_triangles(self.boids.headings,
                                        self.boids.positions)

    def animate_rerun_step(self, time):
        """
        Standard rerun: we take data of self.repository
        with prefix self.filename and replay it,
        we update the step and the triangles
        """
        self.boids.positions, self.boids.velocities, self.boids.headings = \
            get_positions_velocity_headings(self.repository,
                                            self.filename,
                                            self.step)
        self.step += 1
        self.triangles.update_triangles(self.boids.headings,
                                        self.boids.positions)

    def animate_rerun_DBSCAN_intuition_metric(self, time, phi=100, alpha=1.2):
        """
        Rerun DBSCAN algorithm with clusters results from DBSCAN with "intuition metrics"
        """
        self.boids.positions, self.boids.velocities, self.boids.headings = \
            get_positions_velocity_headings(self.repository,
                                            self.filename,
                                            self.step)

        labels = np.loadtxt(self.path + "DBSCAN_intuition_dist_phi="
                            + str(phi) + "_alpha=" + str(alpha) + "_label" + str(self.step))

        self.step += 1

        colors = labels_to_colorlist(np.array(labels, dtype=int))
        self.triangles.set_colors(colors)
        self.triangles.update_triangles(self.boids.headings,
                                        self.boids.positions)

    def animate_label_prop(self, time):
        self.boids.positions, self.boids.velocities, self.boids.headings = \
            get_positions_velocity_headings(self.repository,
                                            self.filename,
                                            self.step)

        list_color = graph_step(self.step,
                                repository=self.repository,
                                filename=self.filename)

        self.step += 1

        self.triangles.set_colors(list_color)
        self.triangles.update_triangles(self.boids.headings,
                                        self.boids.positions)

    def animate_labels_the_data(self, time):
        self.boids.positions, self.boids.velocities, self.boids.headings = \
            get_positions_velocity_headings(self.repository,
                                            self.filename,
                                            self.step)

        self.old_labels = build_ground_truth(self.step, self.old_labels, self.repository,
                                             self.filename, self.list_num_boids)

        color_list = labels_to_colorlist(self.old_labels)

        self.triangles.set_colors(color_list)
        self.triangles.update_triangles(self.boids.headings,
                                        self.boids.positions)

        self.step += 1

    def animate_DBscan_positions(self, time):
        self.boids.positions, self.boids.velocities, self.boids.headings = \
            get_positions_velocity_headings(self.repository,
                                            self.filename,
                                            self.step)

        self.old_labels = DBscan_step_positions(self.step,
                                                self.old_labels,
                                                self.repository,
                                                self.filename)

        color_list = labels_to_colorlist(self.old_labels)

        self.triangles.set_colors(color_list)
        self.triangles.update_triangles(self.boids.headings,
                                        self.boids.positions)

        self.step += 1

    def animate_DBscan_positions_and_velocities(self, time):
        self.boids.positions, self.boids.velocities, self.boids.headings = \
            get_positions_velocity_headings(self.repository,
                                            self.filename,
                                            self.step)
        self.old_labels = DBscan_step_positions_and_velocity(self.step,
                                                             self.old_labels,
                                                             self.repository,
                                                             self.filename)

        color_list = labels_to_colorlist(self.old_labels)

        self.triangles.set_colors(color_list)
        self.triangles.update_triangles(self.boids.headings,
                                        self.boids.positions)

        self.step += 1

    def animate_DBscan_intuition_metric(self, time):
        self.boids.positions, self.boids.velocities, self.boids.headings = \
            get_positions_velocity_headings(self.repository, self.filename,
                                            self.step)

        self.old_labels = DBscan_step_intuition_dist(self.step,
                                                     self.old_labels,
                                                     self.repository,
                                                     self.filename)

        color_list = labels_to_colorlist(self.old_labels)

        self.triangles.set_colors(color_list)
        self.triangles.update_triangles(self.boids.headings,
                                        self.boids.positions)

        self.step += 1

    def animate_DBscan_intuition_metric_multistep(self, time, steps=3):
        for i in range(self.step, self.step + steps):
            self.boids.positions, self.boids.velocities, self.boids.headings = \
                get_positions_velocity_headings(self.repository, self.filename,
                                                self.step)

        self.old_labels = DBscan_step_intuition_dist_multistep(self.step,
                                                               self.old_labels,
                                                               self.repository,
                                                               self.filename, nb_step=steps)

        color_list = labels_to_colorlist(self.old_labels)

        self.triangles.set_colors(color_list)
        self.triangles.update_triangles(self.boids.headings,
                                        self.boids.positions)

        self.step += 1


def main(mode, repository, list_num_boids=[40, 49, 40, 30], positions_init=None, velocities_init=None):
    """
    main function create a simulation with some parameters specified and play a scenario, depending on mode value,
    we select an Simulation.animate_****** type method

    :param mode: mode of the
    :param repository: (optional) min speed for all boids, 0.2 by default.
    """

    # initiate the window
    window = pyglet.window.Window(WINDOW_SIZE[0],
                                  WINDOW_SIZE[1],
                                  "Boids",
                                  resizable=False)
    # set background color
    GL.glClearColor(1, 1, 1, 1)

    # initiate some parameters
    app = None
    func_animate = None
    step_to_begin_for_demo = 305

    # play the scenario depending on :mode: value
    if mode == 0:

        # if we passed specific velocities/positions
        if positions_init is None:

            app = Simulation(list_num_boids=list_num_boids,
                             repository=repository)
            func_animate = app.animate
            if not is_empty(repository):
                EraseFile(repository)
        else:

            app = Simulation(list_num_boids=list_num_boids,
                             repository=repository, velocities_init=velocities_init,
                             positions_init=positions_init)
            func_animate = app.animate
            if not is_empty(repository):
                EraseFile(repository)

    elif mode == 1:

        # simply rerun simulation
        app = Simulation(list_num_boids=list_num_boids,
                         repository=repository,
                         step=step_to_begin_for_demo)
        func_animate = app.animate_rerun_step

    elif mode == 2:

        # use DBSCAN algorithm to cluster the boids with position data,
        # use default metric of DBSCAN
        app = Simulation(list_num_boids=list_num_boids,
                         repository=repository,
                         step=step_to_begin_for_demo)
        func_animate = app.animate_DBscan_positions

    elif mode == 3:

        # use DBSCAN algorithm to cluster the boids with position
        # and velocities data,
        # use default metric of DBSCAN
        app = Simulation(list_num_boids=list_num_boids,
                         repository=repository,
                         step=step_to_begin_for_demo)
        func_animate = app.animate_DBscan_positions_and_velocities

    elif mode == 5:
        # use label propagation to cluster the boids
        # /!\ very slow because we build subgraphs and apply
        # label prop onto these subgraphs
        app = Simulation(list_num_boids=list_num_boids,
                         repository=repository,
                         step=step_to_begin_for_demo)
        func_animate = app.animate_label_prop

    elif mode == 7:
        # use species information and DBSCAN to build what we call
        # our "ground truth"
        app = Simulation(list_num_boids=list_num_boids,
                         repository=repository,
                         step=step_to_begin_for_demo)

        func_animate = app.animate_labels_the_data

    elif mode == 8:
        # use a specific metric on DBSCAN, slower than built-in metric
        app = Simulation(list_num_boids=list_num_boids,
                         repository=repository,
                         step=step_to_begin_for_demo)

        func_animate = app.animate_DBscan_intuition_metric

    elif mode == 9:
        # use a specific metric on DBSCAN, much slower than built-in metric
        app = Simulation(list_num_boids=list_num_boids,
                         repository=repository,
                         step=step_to_begin_for_demo)

        func_animate = app.animate_DBscan_intuition_metric_multistep

    elif mode == 10:
        # use a specific metric on DBSCAN, much slower than built-in metric
        app = Simulation(list_num_boids=list_num_boids,
                         repository=repository,
                         step=step_to_begin_for_demo)

        func_animate = app.animate_rerun_DBSCAN_intuition_metric

    @window.event
    def on_draw():
        """
        clear the window and draw the triangles
        """
        window.clear()
        app.triangles.draw_triangles()

    pyglet.clock.schedule_interval(func_animate, 1 / 40)
    pyglet.app.run()


if __name__ == "__main__":

    # choose a mode
    mode = 0

    main(mode=mode, repository="simulation_data/", list_num_boids=[30, 30, 30, 30])
