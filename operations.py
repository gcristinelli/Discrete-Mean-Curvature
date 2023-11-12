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


def create_mesh(coord, n, random, d, rd):
    if random and d == 2:
        here = Rectangle(Point(coord[0][0], coord[0][1]), Point(coord[1][0], coord[1][1]))
        domain = generate_mesh(here, n[3])
        plot(domain, linewidth=0.25)
        pp.savefig(rd + '/mesh.png', bbox_inches='tight', dpi=300)
        pp.close()
    elif random and d == 3:
        here = Box(Point(coord[0][0], coord[0][1], coord[0][2]),
                   Point(coord[1][0], coord[1][1], coord[1][2]))
        domain = generate_mesh(here, n[3])
    elif not random and d == 2:
        domain = RectangleMesh(Point(coord[0][0], coord[0][1]), Point(coord[1][0], coord[1][1]),
                               n[0], n[1], 'crossed')
        plot(domain, linewidth=0.25)
        pp.savefig(rd + '/mesh.png', bbox_inches='tight', dpi=300)
        pp.close()
    else:
        domain = BoxMesh(Point(coord[0][0], coord[0][1], coord[0][1]),
                         Point(coord[1][0], coord[1][1], coord[1][2]), n[0], n[1], n[2])
    return domain


def linear_problem(domain, graph, face_coeff, dist, cut_result, vol_cells, bdy_length):
    start_time = time.time()
    val = face_coeff * dist.vector() * vol_cells.vector() + bdy_length.vector()
    graph.add_grid_tedges(np.arange(domain.num_cells()), np.maximum(0.0, val), np.maximum(0.0, -val))
    energy = graph.maxflow()
    cut_result.vector()[:] = graph.get_grid_segments(np.arange(domain.num_cells())).astype(float)
    graph.add_grid_tedges(np.arange(domain.num_cells()), -np.maximum(0.0, val), -np.maximum(0.0, -val))
    print(" cut took - %.2f seconds" % (time.time() - start_time))
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
    fun.vector()[:] = 2 * (fun.vector()[:] - 0.5)
    fun = interpolate(fun, vr)
    dist.vector()[:] = - fun.vector().get_local() * dist.vector().get_local()
    print("Computing the distance took - {} seconds \n".format(time.time() - start))
    return dist


def plot_result(domain, adj_cells, rd, f, n, j, d):
    if d == 2:
        ax = plot(f)
        pp.colorbar(ax, shrink=0.55, format='%01.3f')
        if n == 0: pp.savefig(rd + '/input.png', bbox_inches='tight', dpi=600)
        if n == 1: pp.savefig(rd + '/cut_%s.png' % j, bbox_inches='tight', dpi=600)
        if n == 2: pp.savefig(rd + '/sdist_%s.png' % j, bbox_inches='tight', dpi=600)
        pp.close()
    elif n != 2:
        _export_ply(domain, adj_cells, f, j, n, 1 / 255, rd)


def _export_ply(domain, adj_cells, fun, j, index, threshold, rd):
    domain.init(2, 0)
    f2v = domain.topology()(2, 0)
    var = np.abs(fun.vector()[adj_cells[:, 2]] - fun.vector()[adj_cells[:, 1]])
    var_index = np.column_stack((adj_cells[:, 0], var))
    non_zero_var = var_index[var_index[:, 1] > threshold]
    normalized_var = (255 * (non_zero_var[:, 1] - np.min(non_zero_var[:, 1])) / (np.max(non_zero_var[:, 1]))).astype(
        int)
    n = len(non_zero_var[:, 0])
    header = "ply\nformat ascii 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nelement " \
             "face {}\nproperty list uchar int vertex_indices\nproperty uchar red \nproperty uchar green \nproperty " \
             "uchar blue \n end_header\n"
    if index == 0:
        f = open(rd + '/input.ply', "w")
        f.write(header.format(3 * n, n))
    elif index == 1:
        f = open(rd + '/cut_%s.ply' % j, "w")
        f.write(header.format(3 * n, n))

    triangles = np.arange(3 * n, dtype=int)
    vertices_per_face = 3 * np.ones((n, 1), dtype=int)
    triangles = np.concatenate((vertices_per_face, triangles.reshape((-1, 3))), axis=1)

    for face_index in range(n):
        vert = f2v(int(non_zero_var[face_index, 0]))
        for vertex in vert:
            f.write(" ".join(str(coord) for coord in domain.coordinates()[vertex]) + "\n")

    for face_index in range(n):
        f.write(" ".join(str(ind) for ind in triangles[face_index, :]) + " " + " ".join(
            str(normalized_var[face_index]) for ind in range(3)) + "\n")

    f.close()
