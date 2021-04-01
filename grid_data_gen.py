import matplotlib.pyplot as plt
import numpy as np
from utils import get_positions_velocity_headings
from matplotlib import animation

WINDOW_SIZE = 2000
GRID_SIZE = 1000


def positions_to_grid(positions):

    grid = np.zeros((GRID_SIZE, GRID_SIZE))

    divisor = int(WINDOW_SIZE/GRID_SIZE)

    positions = positions/divisor
    for i in range(positions.shape[0]):

        grid[int(positions[i, 0]), int(positions[i, 1])] = 1

    return grid


def get_positions(repository, filename, step):

    positions, _, _ = get_positions_velocity_headings(repository, filename, step)
    return positions


def step_():
    global repository, filename, step
    positions = get_positions(repository, filename, step)
    grid = positions_to_grid(positions)
    step += 1
    return grid


def update():
    pass


def data_gen():
    while True:
        yield step()


if __name__ == "__main__":

    step = 500
    end = 1000
    repository = "simulation_data/"
    filename = "file"

    for i in list(np.arange(step, end)):
        grid = step_()
        print(grid)


    """
    fig, ax = plt.subplots()
    mat = ax.matshow(generate_data())
    plt.colorbar(mat)
    ani = animation.FuncAnimation(fig, update, data_gen, interval=200,
                                  save_count=50)
    plt.show()
    """