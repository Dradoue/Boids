import math
import numpy as np
import pyglet

from constants import DEFAULT_SIZE, TRIANGLES_COLORS
# imports from local files
from utils import generate_aleatory_positions, \
    generate_aleatory_angles


class Triangles:
    """
    Triangles class

    used for efficient construction and update
    of triangles displaying on the window

    list_num_triangles: number of each
    type of triangles you want to draw
    Triangles types will be drawn with
    same size but different colors by default

    :param kwargs:

    color_list: list containing colors for each triangles (optional)
    width_list: widths of triangles (optional)
    length_list: lengths of triangles (optional)
    list_positions: list of tuple with positions in float
    list_headings: list of angles in degree
    """

    def __init__(self, list_num_triangles=[50, 50], **kwargs):

        # if lists arguments are specified, verification of conditions
        # if 'clé' in dict:
        #   pass
        self.list_num_triangles = list_num_triangles

        self.num_type_triangles = len(self.list_num_triangles)

        self.num_triangles = np.sum(self.list_num_triangles)

        self.list_positions = \
            kwargs.get('list_positions',
                       generate_aleatory_positions(self.num_triangles))

        self.list_heading = \
            kwargs.get('list_headings',
                       generate_aleatory_angles(self.num_triangles))

        self.width_list = \
            kwargs.get('width_list',
                       [kwargs.get('width', DEFAULT_SIZE)]
                       * len(self.list_num_triangles))

        self.length_list = \
            kwargs.get('length_list',
                       [kwargs.get('length', DEFAULT_SIZE * 2)]
                       * len(self.list_num_triangles))

        self.color_list = kwargs.get('color_list', self.init_color_list()
                                     )

        # default positioning of vertices for the design and positioning
        # of triangles
        self.list_offsets = []
        self.init_offset()

        # where all the vertices will be stored within the batch
        self.list_all_vertices = [None] * len(self.list_positions)

        self.batch = pyglet.graphics.Batch()
        self.init_vertices()
        self.update_vertices()

    def init_color_list(self):
        """
        Initialise list of colors
        """
        list_color = []
        for nb_tri in range(len(self.list_num_triangles)):
            for _ in range(self.list_num_triangles[nb_tri]):
                list_color.append(tuple(TRIANGLES_COLORS[nb_tri] * 3))

        return list_color

    def init_offset(self):
        """
        initialise offsets of triangles
        """
        for i in range(self.num_type_triangles):
            self.list_offsets.append(((self.length_list[i] / 2, 0),
                                      (-self.length_list[i] / 2,
                                       -self.width_list[i] / 2),
                                      (-self.length_list[i] / 2,
                                       self.width_list[i] / 2)))

    def init_vertices(self):
        """
        initialise the list of vertices
        """
        for i in range(np.sum(self.list_num_triangles)):
            self.list_all_vertices[i] = \
                self.batch.add(3,
                               pyglet.gl.GL_TRIANGLES,
                               None,
                               ('v2f', [0, 0, 400, 50, 200, 300]),
                               # self.get_vertices_i(i, self.list_offsets[0])
                               ('c3B', self.color_list[i])  # self.color_list[i]
                               )

    def update_vertices(self):
        """
        update the vertices thanks to positions and headings
        """
        for i in range(self.num_triangles):
            new_vertices = self.get_vertices_i(i, self.list_offsets[0])
            for j in range(6):
                self.list_all_vertices[i].vertices[j] = new_vertices[j]

    def set_colors(self, color_list):
        """
        update the colors of triangles with color_list
        """
        for i in range(self.num_triangles):
            self.list_all_vertices[i].colors = tuple(color_list[i] * 3)

    def get_vertices_i(self, i, offset):
        """
        Gets the vertices for use in vertex list
        :return [(x1,y1),(x2,y2),(x3,y3)]: list of vertices
        """
        vertices = []

        for x_offset, y_offset in offset:
            vertices.append(self.list_positions[i][0]
                            + x_offset * math.cos(self.list_heading[i])
                            - (y_offset * math.sin(self.list_heading[i])))
            vertices.append(self.list_positions[i][1]
                            + x_offset * math.sin(self.list_heading[i])
                            + (y_offset * math.cos(self.list_heading[i])))
        return vertices

    def draw_triangles(self):
        """
        Draw the triangles on the screen
        """
        self.batch.draw()

    def set_positions(self, new_positions):
        """
        update the positions
        """
        self.list_positions = new_positions

    def set_headings(self, new_headings):
        """
        update heading of each boid
        """
        self.list_heading = new_headings

    def update_triangles(self, new_headings, new_positions):

        self.set_headings(new_headings)
        self.set_positions(new_positions)
        self.update_vertices()
