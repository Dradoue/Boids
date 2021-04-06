import os
import pyglet
import pyglet.gl as GL
from Simulation import Simulation
from utils import is_empty, erase_files, write_constants_into_simulation_directory
import os

from constants import WINDOW_SIZE


def main(mode, repository, list_num_boids, positions_init=None, velocities_init=None, step_to_begin=300):
    """
    main function create a simulation with some parameters specified and play a scenario, depending on mode value,
    we select an Simulation.animate_****** type method

    :param mode: mode of the
    :param repository: where the data will be stored
    :param list_num_boids: list with number of each boids species
    :param positions_init: if None we will use default initialisation (useful only when mode=0)
    :param velocities_init: if None we will use default initialisation (useful only when mode=0)
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

    # play the scenario depending on :mode: value
    if mode == 0:

        # if we passed specific init velocities/positions
        if positions_init is None:

            app = Simulation(list_num_boids=list_num_boids,
                             repository=repository)
            func_animate = app.animate

            if os.path.exists("data/" + repository):

                if not is_empty("data/" + repository):

                    erase_files("data/" + repository)
            else:
                print("local directory " + "data/" + repository + " does not exist")
                print("creating new directory")
                os.mkdir("data/" + repository)

        else:

            app = Simulation(list_num_boids=list_num_boids,
                             repository=repository, velocities_init=velocities_init,
                             positions_init=positions_init)
            func_animate = app.animate

            if os.path.exists("data/" + repository):

                if not is_empty("data/" + repository):
                    erase_files("data/" + repository)
            else:
                print("local directory " + "data/" + repository + " does not exist")
                print("creating new directory")
                os.mkdir("data/" + repository)

        write_constants_into_simulation_directory(repository)

    elif mode == 1:

        if os.path.exists("data/" + repository):
            # simply rerun simulation
            app = Simulation(list_num_boids=list_num_boids,
                             repository=repository,
                             step=step_to_begin)
            func_animate = app.animate_rerun_step

        else:
            print("local directory " + "data/" + repository + " does not exist")
            exit()

    elif mode == 2:

        if os.path.exists("data/" + repository):
            # use DBSCAN algorithm to cluster the boids with position data,
            # use default metric of DBSCAN
            app = Simulation(list_num_boids=list_num_boids,
                             repository=repository,
                             step=step_to_begin)
            func_animate = app.animate_DBscan_positions
        else:
            print("local directory " + "data/" + repository + " does not exist")
            exit()

    elif mode == 3:

        if os.path.exists("data/" + repository):

            # use DBSCAN algorithm to cluster the boids with position
            # and velocities data,
            # use default metric of DBSCAN
            app = Simulation(list_num_boids=list_num_boids,
                             repository=repository,
                             step=step_to_begin)
            func_animate = app.animate_DBscan_positions_and_velocities
        else:
            print("local directory " + "data/" + repository + " does not exist")
            exit()

    elif mode == 4:

        if os.path.exists("data/" + repository):
            # use label propagation to cluster the boids
            # /!\ very slow because we build subgraphs and apply
            # label prop onto these subgraphs
            app = Simulation(list_num_boids=list_num_boids,
                             repository=repository,
                             step=step_to_begin)
            func_animate = app.animate_label_prop
        else:
            print("local directory " + "data/" + repository + " does not exist")
            exit()

    elif mode == 5:

        if os.path.exists("data/" + repository):
            # use species information and DBSCAN to build what we call
            # our "ground truth"
            app = Simulation(list_num_boids=list_num_boids,
                             repository=repository,
                             step=step_to_begin)

            func_animate = app.animate_labels_the_data
        else:
            print("local directory " + "data/" + repository + " does not exist")
            exit()

    elif mode == 6:

        if os.path.exists("data/" + repository):

            # use a specific metric on DBSCAN, slower than built-in metric
            app = Simulation(list_num_boids=list_num_boids,
                             repository=repository,
                             step=step_to_begin)

            func_animate = app.animate_DBscan_intuition_metric
        else:
            print("local directory " + "data/" + repository + " does not exist")
            exit()

    elif mode == 7:

        if os.path.exists("data/" + repository):

            # use a specific metric on DBSCAN on multiple time-steps, much slower than built-in metric
            app = Simulation(list_num_boids=list_num_boids,
                             repository=repository,
                             step=step_to_begin)

            func_animate = app.animate_DBscan_intuition_metric_multistep
        else:
            print("local directory " + "data/" + repository + " does not exist")
            exit()

    elif mode == 8:

        if os.path.exists("data/" + repository):
            # rerun simulation with specific metric on DBSCAN
            app = Simulation(list_num_boids=list_num_boids,
                             repository=repository,
                             step=step_to_begin)

            func_animate = app.animate_rerun_DBSCAN_intuition_metric
        else:
            print("local directory " + "data/" + repository + " does not exist")
            exit()

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
    mode = 0  # choose mode from 0 to 8, see behind in *main* function
    repository = "simulation_data_new/"  # where the data will be stored in \data\*repository*
    list_num_boids = [30, 30, 30, 30]  # number of boids for each species
    step_to_begin = 300  # step where the rerun-simulation begin, useless for mode=0 where we begin
    # a new simulation and begin at step=0

    main(mode=mode, repository=repository, list_num_boids=list_num_boids, step_to_begin=step_to_begin)
