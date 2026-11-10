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
    input_val = (sl * (np.abs(x[0] - coord[0][0])) + (x[1] - coord[1][1]) < 0.0) | ((x[0] - coord[1][0]) ** 2 + (
                        x[1] - (coord[0][1] + 0.5 * (coord[1][1] - coord[0][1]))) ** 2 < np.minimum(
                    0.5 * (coord[1][0] - coord[0][0]), 0.5 * (coord[1][1] - coord[0][1])) ** 2)
    return float(input_val)


def input_cylinder(x, coord, height, d):
    """
    Evaluate the input function for a cylinder.

    Parameters:
    - x: Coordinates of the point.
    - coord: Lower and upper domain corners; the cylinder starts at the lower axial bound.
    - height: Height of the cylinder.
    - d: Spatial dimension.

    Returns:
    - float: The value of the input function at the given point.
    """
    radius = 0.2*(coord[1][0]-coord[0][0])
    center_x = 0.5 * (coord[0][0] + coord[1][0])
    if d==2:
        input_val = (abs(x[0] - center_x) < height) & (coord[0][1] <= x[1] <= coord[0][1] + radius)
    else:
        center_y = 0.5 * (coord[0][1] + coord[1][1])
        distance_to_center = np.sqrt((x[1] - center_y) ** 2 + (x[0] - center_x) ** 2)
        input_val = (distance_to_center < radius) & (coord[0][2] <= x[2] <= coord[0][2] + height)
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
        domain = BoxMesh(Point(coord[0][0], coord[0][1], coord[0][2]),
                         Point(coord[1][0], coord[1][1], coord[1][2]), n[0], n[1], n[2])
    return domain


def linear_problem(cells_list, graph, face_coeff, dist, cut_result, vol_cells, bdy_length,
                   adj_cells, facet_size, target_volume, volume_coeff, max_volume_it=32):
    """Search graph cuts for E + volume_coeff / 2 * (volume - target_volume)**2.

    The multiplier satisfies lambda = volume_coeff * (volume - target_volume).
    Binary cut volumes can jump, so keep the best penalized candidate, including
    the previous cut. This is not a global solver for the binary soft constraint.
    The returned energy includes signed unary terms and the volume penalty.
    """
    start_time = time.time()
    volumes = vol_cells.vector().get_local()
    total_volume = volumes.sum()
    if not np.isfinite(volume_coeff) or volume_coeff < 0:
        raise ValueError("volume_coeff must be finite and nonnegative")
    if not np.isfinite(target_volume) or not 0 <= target_volume <= total_volume:
        raise ValueError("target_volume must lie between zero and the domain volume")
    val = face_coeff * dist.vector().get_local() * volumes + bdy_length.vector().get_local()
    weights = facet_size[adj_cells[:, 0]]

    def penalized_energy(labels):
        volume = np.dot(volumes, labels)
        perimeter = np.dot(weights, np.abs(labels[adj_cells[:, 1]] - labels[adj_cells[:, 2]]))
        energy = perimeter + np.dot(val, labels) + 0.5 * volume_coeff * (volume - target_volume)**2
        return energy, volume

    best_labels = cut_result.vector().get_local()
    best_energy, best_volume = penalized_energy(best_labels)
    lower = -volume_coeff * target_volume
    upper = volume_coeff * (total_volume - target_volume)
    multiplier = 0.0
    # Include the bracket endpoints, then bisect the monotone multiplier residual.
    multipliers = [0.0, lower, upper] if volume_coeff else [0.0]
    solves = 0
    for iteration in range(len(multipliers) + max_volume_it if volume_coeff else 1):
        if iteration < len(multipliers):
            multiplier = multipliers[iteration]
        else:
            multiplier = 0.5 * (lower + upper)
        unary = val + multiplier * volumes
        positive = np.maximum(0.0, unary)
        negative = np.maximum(0.0, -unary)
        graph.add_grid_tedges(cells_list, positive, negative)
        try:
            graph.maxflow()
            labels = graph.get_grid_segments(cells_list).astype(float)
        finally:
            graph.add_grid_tedges(cells_list, -positive, -negative)
        solves += 1
        energy, volume = penalized_energy(labels)
        if energy < best_energy or not volume_coeff:
            best_labels, best_energy, best_volume = labels, energy, volume
        residual = multiplier - volume_coeff * (volume - target_volume)
        if not volume_coeff or residual == 0:
            break
        if residual < 0:
            lower = max(lower, multiplier)
        else:
            upper = min(upper, multiplier)
        if iteration >= len(multipliers) - 1 and upper - lower <= 1e-8 * max(1.0, abs(lower), abs(upper)):
            break
    cut_result.vector()[:] = best_labels
    print(" cut took - {:.2f} seconds ({} graph cuts)".format(time.time() - start_time, solves))
    print(" volume = {:.8g}, target = {:.8g}, error = {:+.3e}".format(best_volume, target_volume, best_volume - target_volume))
    return best_energy


def extract_bdy_nodes(fun, facet_nodes, bdy_facets, bdy_cells, adj_cells):
    values = fun.vector().get_local()
    bdy_fun_indices = np.concatenate((
        bdy_facets[values[bdy_cells] != 0],
        adj_cells[values[adj_cells[:, 1]] != values[adj_cells[:, 2]], 0],
    ))
    return np.unique(facet_nodes[bdy_fun_indices])


def signed_distance(fun, domain, vr, d2v, source, receiver, slowness):
    start = time.time()
    tt = domain.raytrace(source, receiver, slowness, aggregate_src=True)
    min_distance = domain.get_grid_traveltimes().astype(np.float64)
    dist = Function(vr)
    dist.vector()[:] = min_distance[d2v]
    fun = interpolate(fun, vr)
    fun.vector()[:] = 2 * (fun.vector()[:] - 0.5)
    dist.vector()[:] = - fun.vector().get_local() * dist.vector().get_local()
    print("Computing the distance took - {} seconds \n".format(time.time() - start))
    return dist


def plot_result(domain, adj_cells, facet_nodes, rd, f, n, j, d):
    if d == 2:
        ax = plot(f)
        pp.colorbar(ax, shrink=0.55, format='%01.3f')
        if n == 0: pp.savefig(rd + '/input.png', bbox_inches='tight', dpi=600)
        if n == 1: pp.savefig(rd + '/cut_%s.png' % j, bbox_inches='tight', dpi=600)
        if n == 2: pp.savefig(rd + '/sdist_%s.png' % j, bbox_inches='tight', dpi=600)
        pp.close()
    elif n != 2:
        _export_ply(domain, adj_cells, facet_nodes, f, j, n, 1 / 255, rd)


def _export_ply(domain, adj_cells, facet_nodes, fun, j, index, threshold, rd):
    coordinates = domain.coordinates()
    values = fun.vector().get_local()
    var = np.abs(values[adj_cells[:, 2]] - values[adj_cells[:, 1]])
    var_index = np.column_stack((adj_cells[:, 0], var))
    non_zero_var = var_index[var_index[:, 1] > threshold]
    n = len(non_zero_var[:, 0])
    normalized_var = np.empty(n, dtype=int)
    if n:
        normalized_var[:] = (255 * (non_zero_var[:, 1] - np.min(non_zero_var[:, 1])) / np.max(non_zero_var[:, 1])).astype(int)
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
        vert = facet_nodes[int(non_zero_var[face_index, 0])]
        for vertex in vert:
            f.write(" ".join(str(coord) for coord in coordinates[vertex]) + "\n")

    for face_index in range(n):
        f.write(" ".join(str(ind) for ind in triangles[face_index, :]) + " " + " ".join(
            str(normalized_var[face_index]) for ind in range(3)) + "\n")

    f.close()
