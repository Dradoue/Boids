import os

import numpy as np

from Simulation import Simulation
from utils import is_empty, erase_files

if __name__ == "__main__":

    # step to begin and step to end for all the simulations
    step_to_begin = 0
    step_to_end = 5000

    # list of number of boids, must be same size than list_directories var
    list_num_boids = [[200] * 4, [500] * 4, [1000] * 4]

    # number of simulation to generate for each population
    num_run = 10

    # data of simulation number :i: with a population of :num_boids:
    # will be stored in /data/simulation_data_:num_boids:_Boids_:i:

    for num_boids in list_num_boids:

        for run in range(num_run):

            directory = "simulation_data_" + str(num_boids) + "_Boids_" + str(run) + "/"
            app = Simulation(list_num_boids=num_boids,
                             repository=directory,
                             step=step_to_begin)

            if os.path.exists("data/" + directory):

                if not is_empty("data/" + directory):
                    erase_files("data/" + directory)
            else:
                print("local directory " + "data/" + directory + " does not exist")
                print("creating new directory")
                os.mkdir("data/" + directory)

            # set init step
            app.step = step_to_begin

            for i in np.arange(step_to_begin, step_to_end, 1):
                app.animate(i)
