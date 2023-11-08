from dolfin import *
import maxflow
import time as time
import ufl
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
    nx, ny, n = [500, 300, 400]
    lx1, lx2 = [-0.5, 0.5]
    ly1, ly2 = [0.0, 0.5]
    lz1, lz2 = [0.0, 0.5]
    random = True
    d = 2
    height = 0.2
    input_slope = 1

    # BOUNDARY PARAMETERS
    face_coeff = 30
    bdy_coeff_bone = -0.95
    bdy_coeff_scaffold = -0.9
    bdy_coeff_side = 1

    # ELASTICITY Parameters
    mu = 8.0  # Lame coefficient
    lbd = 16.0  # Lame coefficient
    eps1 = 0.0001  # small parameter avoiding division by zero in the normal computation
    weak = 0.01  # multiplicative coefficient for the "weak" material mimicking void
    e0 = 0.9  # spontaneous strain along bottom boundary
    elas_coeff = 0.00  # coefficient in front of shape derivative for descent

    # GEOMETRY LOOP ----------------------------------------------------------------------------------------------------
    print("Starting geometry loop...\n")
    start = time.time()

    # B1.---MESH GENERATION AND FUNCTION SPACES
    coord = [[lx1, ly1, lz1], [lx2, ly2, lz2]]
    domain = create_mesh(coord, [nx, ny, n], random)
    plot(domain, linewidth=0.25)
    pp.savefig(rd + '/mesh.png', bbox_inches='tight', dpi=300)
    pp.close()

    V = FunctionSpace(domain, 'DG', 0)  # PWC
    VL = FunctionSpace(domain, 'CG', 1)  # PWL
    V_vec = VectorFunctionSpace(domain, 'DG', 0)
    Vmat = TensorFunctionSpace(domain, 'DG', 0)
    VL_vec = VectorFunctionSpace(domain, 'CG', 1)

    vol_cells, bdy_length = Function(V), Function(V)
    domain.init(d - 1, d)
    f2c = domain.topology()(d - 1, d)
    int_lengths, facet_size = np.empty(0), np.empty(0)

    # Creating two array of indices of (d-1)-dimensional objects (boundary or internal)
    facets_list = np.arange(domain.num_facets())
    bdy_facets = np.array([facet for facet in facets_list if len(f2c(facet)) == 1], dtype=int)
    internal_facets = np.setdiff1d(facets_list, bdy_facets)

    # Doing the same for cells
    cells_list = np.arange(domain.num_facets())
    bdy_cells = np.array([f2c(facet)[0] for facet in bdy_facets], dtype=int)
    internal_cells = np.setdiff1d(cells_list, bdy_cells)
    adj_cells = np.array([[facet, f2c(facet)[0], f2c(facet)[1]] for facet in internal_facets], dtype=int)

    vol_cells.vector()[:] = [Cell(domain, cell).volume() for cell in range(domain.num_cells())]
    mid_cell = [Cell(domain, cell).midpoint().array() for cell in range(0, domain.num_cells())]

    # Computing size of facets
    if d == 2:
        facet_size = np.array([Edge(domain, edge).length() for edge in facets_list])
    elif d == 3:
        facet_size = np.array([Face(domain, face).area() for face in facets_list])

    # Adding boundary info
    for facet in bdy_facets:
        if mid_cell[f2c(facet)[0]][d - 1] >= height:
            bdy_length.vector()[f2c(facet)[0]] = +facet_size[facet]

    # B2.---GRAPH GENERATION, creating graph with (d-1)-facets areas/length as weights
    G = maxflow.GraphFloat()
    G.add_nodes(domain.num_cells())
    for facet in internal_facets: G.add_edge(f2c(facet)[0], f2c(facet)[1], facet_size[facet], facet_size[facet])

    print("Making the mesh of {} vertices, {} {}-dimensional cells, and the graph took - {} seconds \n".format(
        domain.num_vertices(), domain.num_cells(), d, time.time() - start))

    # PDE AND INPUT ----------------------------------------------------------------------------------------------------
    input_data = Function(V)
    input_data.vector()[:] = np.array([input_fun(mid_cell[cell], coord, input_slope)
                                       for cell in range(domain.num_cells())])
    # Plotting the input
    ax = plot(input_data, vmin=-1.0, vmax=1.0)
    pp.savefig(rd + '/input.png', bbox_inches='tight', dpi=300)
    pp.close()

    # distance function to boundary of input, first we define a binary function over the facets of the domain
    # that indicates when a facet is in the boundary of the input_data
    bdy_input = MeshFunction('size_t', domain, d - 1, 0)
    bdy_input.array()[bdy_facets] = np.array([abs(input_data.vector()[f2c(facet)[0]]) for facet in bdy_facets])
    bdy_input.array()[adj_cells[:, 0]] = abs(
        input_data.vector()[adj_cells[:, 1]] - input_data.vector()[adj_cells[:, 2]])

    #Fast marching method to solve Eikonal equation

    # MAIN LOOP --------------------------------------------------------------------------------------------------------
    max_it, it, stop = [50, 1, False]
    cut_result = Function(V)
    while it <= max_it or stop:
        face_coeff = face_coeff * 0.8
        print("--Doing iteration", it)
        # L1.---Distance function to the boundary of the previous cut
        bdy_cut = MeshFunction('size_t', domain, d - 1, 0)

        # L2.--- new cut
        cut_value = linear_problem(domain, G, face_coeff, input_data, cut_result, vol_cells, bdy_length)
        ax = plot(cut_result, vmin=0.0, vmax=1.0)
        pp.savefig(rd + '/cut_%s.png' % it, bbox_inches='tight', dpi=300)
        pp.close()
        it += 1


########################################################################################################################

if __name__ == '__main__':
    _main()
