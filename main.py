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
import random as rnd
from ttcrpy.tmesh import Mesh2d, Mesh3d

from operations import *


# import skfmm  # fast marching for distances (scikit-fmm)


########################################################################################################################
def _main():
    # ---make a timestamped folder to spam images
    rd = result_directory()

    # ---variables
    nx, ny, n = [500, 300, 300]
    lx1, lx2 = [-0.5, 0.5]
    ly1, ly2 = [0.0, 0.5]
    lz1, lz2 = [0.0, 0.5]
    random = True
    d = 2
    height = 0.4
    input_slope = 1

    # BOUNDARY PARAMETERS
    face_coeff = 30

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
    Vr = FunctionSpace(domain, 'CG', 1)  # PWL
    V_vec = VectorFunctionSpace(domain, 'DG', 0)
    Vmat = TensorFunctionSpace(domain, 'DG', 0)
    Vr_vec = VectorFunctionSpace(domain, 'CG', 1)

    vol_cells, bdy_length = Function(V), Function(V)
    domain.init()
    f2c = domain.topology()(d - 1, d)
    c2n = domain.topology()(d, 0)
    int_lengths, facet_size = np.empty(0), np.empty(0)

    # Creating two array of indices of (d-1)-dimensional objects (boundary or internal)
    facets_list = np.arange(domain.num_facets())
    bdy_facets = np.array([facet for facet in facets_list if len(f2c(facet)) == 1], dtype=int)
    internal_facets = np.setdiff1d(facets_list, bdy_facets)

    # Doing the same for cells
    cells_list = np.arange(domain.num_cells())
    cells_construction = np.array([c2n(cell) for cell in cells_list], dtype=int)
    bdy_cells = np.array([f2c(facet)[0] for facet in bdy_facets], dtype=int)
    adj_cells = np.array([[facet, f2c(facet)[0], f2c(facet)[1]] for facet in internal_facets], dtype=int)
    # creating adj_cells takes a few seconds, but it makes the creation of the graph extremely efficient

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
            bdy_length.vector()[f2c(facet)[0]] += facet_size[facet]

    # B2. --MOVING MESH TO MESH2D
    start_0 = time.time()
    if d == 2:
        domain2 = Mesh2d(domain.coordinates(), cells_construction, method='SPM')
    else:
        domain2 = Mesh3d(domain.coordinates(), cells_construction, method='FSM')
    print('exporting took {} seconds \n'.format(time.time()-start_0))

    # B2.---GRAPH GENERATION, creating graph with (d-1)-facets areas/length as weights, takes less than 0.1 seconds
    G = maxflow.GraphFloat()
    G.add_nodes(domain.num_cells())
    G.add_edges(adj_cells[:, 1], adj_cells[:, 2], facet_size[adj_cells[:, 0]], facet_size[adj_cells[:, 0]])

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
    bdy_nodes = extract_bdy_nodes(input_data, domain, d, bdy_facets, adj_cells)
    source = np.reshape(domain.coordinates()[1], (1, -1))
    tt, rays = domain2.raytrace(source, bdy_nodes, return_rays=True)
    print(np.count_nonzero(tt))
    mesh_tt = domain2.get_grid_traveltimes()
    print(np.count_nonzero(mesh_tt))

    #plotting the resulting rays
    fig = pp.figure()
    ax = fig.add_subplot()
    ax.plot(bdy_nodes[:, 0], bdy_nodes[:, 1], ',')

    # add rays for the receivers at the surface
    nRx = bdy_nodes.shape[0]
    for r in rays[:nRx]:
        pp.plot(r[:, 0], r[:, 1], c=[0.5, 0.5, 0.5], lw=0.001)

    pp.show()
    pp.close()

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
