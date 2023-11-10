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
    return - float(input_val)


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
    print(" cut took took - %.2f seconds \n" % (time.time() - start_time))
    return energy


def mat2func(px, py, fn, fn_mat, dofsV_max):
    for dof in range(0, dofsV_max):
        if np.rint(px[dof]) % 2 == .0:
            cx, cy = np.int_(np.rint([px[dof] / 2, py[dof] / 2]))
            fn.vector()[dof] = fn_mat[cy, cx]
        else:
            cx, cy = np.int_(np.floor([px[dof] / 2, py[dof] / 2]))
            fn.vector()[dof] = 0.25 * (fn_mat[cy, cx] + fn_mat[cy + 1, cx] \
                                       + fn_mat[cy, cx + 1] + fn_mat[cy + 1, cx + 1])
    return fn


# ----------------------------------------------------------------------
def func2mat(px, py, fn, fn_mat, dofsV_max):
    fn_array = fn.vector().get_local()
    for dof in range(0, dofsV_max):
        cx, cy = np.int_(np.rint([px[dof] / 2, py[dof] / 2]))
        fn_mat[cy, cx] = fn_array[dof]
    return fn_mat


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
    bdy_fun_nodes = domain.coordinates()[bdy_fun_nodes]
    return bdy_fun_nodes

