import numpy as np

# PARAMETERS FOR SIMULATION
WINDOW_SIZE = 2000, 2000  # size of window
BOUNDS = \
    (0, WINDOW_SIZE[0], 0, WINDOW_SIZE[1])
# bounds of the grid for boids
LEAF_SIZE = 2000
DEFAULT_SIZE = 15  # size of triangles
DEFAULT_TOLERANCE = 1E-7  # tolerance for
# boids when they cross boundaries
EPSILON = 1E-3  # tolerance for 0-divisions

NB_SPECIES = 4
# PARAMETERS FOR BOIDS

# relations between species:
# RELATIONS_SPECIES[i, i] = 0 for all i
# species 1 chase species 2 => RELATIONS_SPECIES[0,1]=1
# species 2 flee species 1 => RELATIONS_SPECIES[1,0]=-1
# neutral relation (ignore everything) => 0
RELATIONS_SPECIES = np.array([[0, 1, 0, 0],
                              [-1, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0]])

SEPARATION_DIST = [35, 55, 75, 95]  # separation distances
BOID_VIEW = [200, 200, 200, 200]  # euclidean distance where boids
# can see other boids, for alignment and cohesion

BOID_VIEW_ANGLE = 260  # view angle of boids
FLEE_FORCE = 40
CHASING_FORCE = 40
SEPARATION_FORCE = 2.5
COHESION_FORCE = 20
ALIGNMENT_FORCE = 35

# help verify conditions for constants dependencies
assert len(SEPARATION_DIST) == len(BOID_VIEW) \
       == NB_SPECIES == RELATIONS_SPECIES.shape[0]\
       == RELATIONS_SPECIES.shape[1], 'constants parameters doesnt match length'

# INITIALISATION OF COLORS
np.random.RandomState(seed=0)
TRIANGLES_COLORS = []
for _ in range(300):
    TRIANGLES_COLORS.append(list(np.random.choice(range(256),
                                                  size=3)))
