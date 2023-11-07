from dolfin import *
from mshr import *
from matplotlib import cm, pyplot as pp
from datetime import datetime
from scipy.ndimage import gaussian_filter
from scipy.ndimage import convolve
import numpy as np, sys, os
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
    val = face_coeff * dist.vector() * vol_cells.vector() + bdy_length.vector()
    graph.add_grid_tedges(np.arange(domain.num_cells()), np.maximum(0.0, val), np.maximum(0.0, -val))
    energy = graph.maxflow()
    cut_result.vector()[:] = graph.get_grid_segments(np.arange(domain.num_cells())).astype(float)
    graph.add_grid_tedges(np.arange(domain.num_cells()), -np.maximum(0.0, val), -np.maximum(0.0, -val))
    return energy

def _solve_elasticity_dirichlet(domain, cut, dx, coord, mu, lbd, weak, e0):
    V = FunctionSpace(domain, 'DG', 0)  # PWC
    V_vec = VectorFunctionSpace(domain, 'DG', 0)

    def left(x, on_boundary): return x[0] < coord[0][0] + DOLFIN_EPS

    def right(x, on_boundary): return x[0] > coord[1][0] - DOLFIN_EPS

    def up(x, on_boundary): return x[1] > coord[0][1] - DOLFIN_EPS

    def down(x, on_boundary): return x[1] < coord[1][1] + DOLFIN_EPS

    mismatch = Expression(("-1.0 * E0 * ( x[0] - 0.5*(Lx2+Lx1) )", " 0.0"), E0=e0, Lx1=coord[0][0], Lx2=coord[1][0],
                          degree=1)
    bc3 = DirichletBC(V_vec, mismatch, down)
    bcs = [bc3]
    u, v = [TrialFunction(V_vec), TestFunction(V_vec)]
    S1 = 2.0 * mu * inner(sym(grad(u)), sym(grad(v))) + lbd * div(u) * div(v)
    a = S1 * weak * dx(0) + S1 * dx(1)
    L = inner(Constant((0.0, 0.0)), v) * dx
    A, b = assemble_system(a, L, bcs)
    solver = LUSolver(A)
    solver.solve(U.vector(), b)
    SE = 2.0 * mu * inner(sym(grad(U)), sym(grad(U))) + lbd * div(U) * div(U)
    energy = assemble(SE * weak * dx(0) + SE * dx(1))
    edens_aux = Function(V)
    edens_aux = project(SE, V)
    edens = Function(V)
    edens_vec = np.multiply(edens_aux.vector(), cut.vector()) + \
                weak * np.multiply(edens_aux.vector(), 1.0 - cut.vector())
    edens.vector()[:] = edens_vec
    return edens, energy
