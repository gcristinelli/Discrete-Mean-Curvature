from dolfin import *
import maxflow
import time as time
from mshr import *
from matplotlib import cm, pyplot as pp
from datetime import datetime
from scipy.ndimage import gaussian_filter
from scipy.ndimage import convolve
import numpy as np, sys, os
import networkx as nx
import random as rnd

from operations import *


# import skfmm  # fast marching for distances (scikit-fmm)


########################################################################################################################
def _main():
    # ---make a timestamped folder to spam images
    rd = result_directory()

    # ---variables
    nx, ny = [300, 200]
    lx1, lx2 = [-0.5, 0.5]
    ly1, ly2 = [0.0, 0.5]
    random = True
    d = 2

    # GEOMETRY LOOP ----------------------------------------------------------------------------------------------------
    print("Starting geometry loop...\n")
    start = time.time()

    # B1.---MESH GENERATION AND FUNCTION SPACES
    domain = create_mesh([[lx1, ly1], [lx2, ly2]], [nx, ny])
    plot(domain, linewidth=0.25)
    pp.savefig(rd + '/mesh.png', bbox_inches='tight', dpi=300)
    pp.close()

    V = FunctionSpace(domain, 'DG', 0)  # PWC
    VL = FunctionSpace(domain, 'CG', 1)  # PWL
    vol_cells = Function(V)
    domain.init()
    domain.init(d - 1, d)
    f2c = domain.topology()(d - 1, d)
    adjacency = np.empty(shape=[0, 2])
    int_lengths, facet_size = np.empty(0), np.empty(0)

    # Creating two array of indices of (d-1)-dimensional objects (boundary or internal)
    facets_list = np.arange(domain.num_facets())
    bdy_facets = np.array([facet for facet in facets_list if len(f2c(facet)) == 1], dtype=int)
    internal_facets = np.setdiff1d(facets_list, bdy_facets)

    # Doing the same for cells
    cells_list = np.arange(domain.num_facets())
    bdy_cells = np.array([f2c(facet)[0] for facet in bdy_facets], dtype=int)
    internal_cells = np.setdiff1d(cells_list, bdy_cells)

    # Computing size of facets
    if d == 2:
        facet_size = np.array([Edge(domain, edge).length() for edge in facets_list])
    elif d == 3:
        facet_size = np.array([Face(domain, face).area() for face in facets_list])

    # B2.---GRAPH GENERATION
    G = maxflow.GraphFloat()
    G.add_nodes(domain.num_cells())

    # Creating graph with (d-1)-facets areas/length as weights
    for facet in internal_facets:
        adj_cells = f2c(facet)
        G.add_edge(adj_cells[0], adj_cells[1], facet_size[facet], facet_size[facet])
        adjacency = np.append(adjacency, np.reshape(adj_cells, (1, -1)), axis=0)
        int_lengths = np.append(int_lengths, facet_size[facet])

    # Creating adjacency matrix for internal facets: the first column gives the index of the internal facet,
    # while the other two give the indices of the adjacent cells
    adjacency = np.concatenate((internal_facets.reshape((-1, 1)), adjacency), axis=1)

    vol_cells.vector()[:] = [Cell(domain, cell).volume() for cell in range(domain.num_cells())]
    mid_cell = [Cell(domain, cell).midpoint().array() for cell in range(0, domain.num_cells())]

    print("Making the mesh of {} vertices, {} {}-dimensional cells, and the graph took - {} seconds \n".format(
        domain.num_vertices(), domain.num_cells(), d, time.time() - start))


########################################################################################################################

if __name__ == '__main__':
    _main()
