from dolfin import *
from mshr import *
from matplotlib import cm, pyplot as pp
from datetime import datetime
from scipy.ndimage import gaussian_filter
from scipy.ndimage import convolve
import numpy as np, sys, os
import time as time
import random as rnd


def result_directory():
    now = datetime.now()
    dt_string = now.strftime("%y%m%d_%H%M%S")
    rd = os.path.join(os.path.dirname(__file__), './results/' + dt_string)
    if not os.path.isdir(rd): os.makedirs(rd)
    return rd


def input_fun(x, coord, sl):
    input_val = (sl * (np.abs(x[0] - coord[0][0])) + (x[1] - coord[1][1]) < 0.0) | \
                ((x[0] - coord[1][0]) ** 2 + (
                        x[1] - (coord[0][1] + 0.5 * (coord[1][1] - coord[0][1]))) ** 2 < np.minimum(
                    0.5 * (coord[1][0] - coord[0][0]), 0.5 * (coord[1][1] - coord[0][1])) ** 2)
    return float(input_val)


def create_mesh(coord, n, random):
    if random:
        here = Rectangle(Point(coord[0][0], coord[0][1]), Point(coord[1][0], coord[1][1]))
        domain = generate_mesh(here, n[2])
    else:
        domain = RectangleMesh(Point(coord[0][0], coord[0][1]), Point(coord[1][0], coord[1][1]),
                               n[0], n[1], 'crossed')
    return domain


def linear_problem(domain, graph, face_coeff, dist, cut_result, vol_cells, bdy_length):
    start_time = time.time()
    val = face_coeff * dist.vector() * vol_cells.vector() + bdy_length.vector()
    graph.add_grid_tedges(np.arange(domain.num_cells()), np.maximum(0.0, val), np.maximum(0.0, -val))
    energy = graph.maxflow()
    cut_result.vector()[:] = graph.get_grid_segments(np.arange(domain.num_cells())).astype(float)
    graph.add_grid_tedges(np.arange(domain.num_cells()), -np.maximum(0.0, val), -np.maximum(0.0, -val))
    print(" cut took - %.2f seconds \n" % (time.time() - start_time))
    return energy


def extract_bdy_nodes(fun, domain, d, bdy_facets, adj_cells):
    f2n = domain.topology()(d - 1, 0)
    f2c = domain.topology()(d - 1, d)
    bdy_fun = MeshFunction('size_t', domain, d - 1, 0)
    bdy_fun.array()[bdy_facets] = np.array([abs(fun.vector()[f2c(facet)[0]]) for facet in bdy_facets])
    bdy_fun.array()[adj_cells[:, 0]] = abs(
        fun.vector()[adj_cells[:, 1]] - fun.vector()[adj_cells[:, 2]])
    bdy_fun_indices = np.column_stack(np.nonzero(bdy_fun.array())).astype(int)
    bdy_fun_nodes = np.array([f2n(facet) for facet in bdy_fun_indices], dtype=int)
    bdy_fun_nodes = np.column_stack(bdy_fun_nodes).astype(int)
    bdy_fun_nodes = np.unique(bdy_fun_nodes)
    return bdy_fun_nodes


def signed_distance(fun, domain, vr, source, receiver, slowness):
    d2v = dof_to_vertex_map(vr)
    start = time.time()
    tt = domain.raytrace(source, receiver, slowness, aggregate_src=True)
    min_distance = domain.get_grid_traveltimes().astype(np.float64)
    dist = Function(vr)
    dist.vector()[:] = np.array([min_distance[d2v[i]] for i in range(domain.get_number_of_nodes())], dtype=np.float64)
    fun.vector()[:] = 2*(fun.vector()[:] - 0.5)
    fun = interpolate(fun, vr)
    dist.vector()[:] = - fun.vector().get_local()*dist.vector().get_local()
    print("Computing the distance took - {} seconds \n".format(time.time() - start))
    return dist
