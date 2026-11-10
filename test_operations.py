import contextlib
import io
import itertools
import unittest

import maxflow
import numpy as np
from dolfin import Function, FunctionSpace, UnitSquareMesh

from operations import linear_problem


class SoftVolumeTests(unittest.TestCase):
    def setUp(self):
        self.space = FunctionSpace(UnitSquareMesh(1, 1), 'DG', 0)
        self.cells = np.arange(2)
        self.adj = np.array([[0, 0, 1]], dtype=int)

    def solve(self, unary, target, coefficient, volumes=(1.0, 1.0), weight=0.0,
              previous=(1.0, 0.0), graph=None, **kwargs):
        dist, cut, cell_volumes, boundary = [Function(self.space) for _ in range(4)]
        cut.vector()[:] = previous
        cell_volumes.vector()[:] = volumes
        boundary.vector()[:] = unary
        if graph is None:
            graph = maxflow.GraphFloat()
            graph.add_nodes(2)
            graph.add_edge(0, 1, weight, weight)
        with contextlib.redirect_stdout(io.StringIO()):
            energy = linear_problem(
                self.cells, graph, 1.0, dist, cut, cell_volumes, boundary,
                self.adj, np.array([weight]), target, coefficient, **kwargs)
        labels = cut.vector().get_local()
        expected_energy = (np.dot(unary, labels) + weight * abs(labels[0] - labels[1])
                           + 0.5 * coefficient * (np.dot(volumes, labels) - target)**2)
        self.assertAlmostEqual(energy, expected_energy)
        return labels, energy, graph

    def assert_global_minimum(self, energy, unary, target, coefficient,
                              volumes=(1.0, 1.0), weight=0.0):
        minimum = min(
            np.dot(unary, labels) + weight * abs(labels[0] - labels[1])
            + 0.5 * coefficient * (np.dot(volumes, labels) - target)**2
            for labels in itertools.product((0.0, 1.0), repeat=2))
        self.assertAlmostEqual(energy, minimum)

    def test_zero_coefficient_recovers_unconstrained_cut(self):
        labels, energy, _ = self.solve((-2.0, 1.0), 1.0, 0.0, weight=0.25)
        np.testing.assert_array_equal(labels, (1.0, 0.0))
        self.assert_global_minimum(energy, (-2.0, 1.0), 1.0, 0.0, weight=0.25)

    def test_penalty_is_soft_and_strength_is_adjustable(self):
        for coefficient, expected in [(0.1, (1.0, 1.0)), (1e6, (1.0, 0.0))]:
            labels, energy, _ = self.solve((-4.0, -0.1), 1.0, coefficient)
            np.testing.assert_array_equal(labels, expected)
            self.assert_global_minimum(energy, (-4.0, -0.1), 1.0, coefficient)

    def test_search_finds_nontrivial_cut_with_unequal_volumes(self):
        labels, energy, _ = self.solve(
            (-3.0, -2.0), 1.0, 1000.0, volumes=(1.0, 2.0), previous=(0.0, 0.0))
        np.testing.assert_array_equal(labels, (1.0, 0.0))
        self.assert_global_minimum(energy, (-3.0, -2.0), 1.0, 1000.0, volumes=(1.0, 2.0))

    def test_empty_and_full_target_volumes(self):
        for target, expected in [(0.0, (0.0, 0.0)), (3.0, (1.0, 1.0))]:
            labels, energy, _ = self.solve(
                (-1.0, 0.5), target, 1e5, volumes=(1.0, 2.0), weight=0.3)
            np.testing.assert_array_equal(labels, expected)
            self.assert_global_minimum(
                energy, (-1.0, 0.5), target, 1e5, volumes=(1.0, 2.0), weight=0.3)

    def test_volume_jump_retains_better_previous_cut(self):
        # Linear-volume cuts cannot select just one of these identical cells.
        labels, energy, _ = self.solve(
            (0.0, 0.0), 1.0, 1e6, weight=1.0, max_volume_it=4)
        np.testing.assert_array_equal(labels, (1.0, 0.0))
        self.assertEqual(energy, 1.0)

    def test_graph_reuse_matches_fresh_graph(self):
        graph = None
        previous = (1.0, 0.0)
        for unary in [(-3.0, -2.0), (4.0, -1.0), (-0.1, -0.2), (1.0, 3.0)]:
            labels, energy, graph = self.solve(
                unary, 1.0, 500.0, volumes=(1.0, 2.0), weight=0.7,
                previous=previous, graph=graph)
            fresh_labels, fresh_energy, _ = self.solve(
                unary, 1.0, 500.0, volumes=(1.0, 2.0), weight=0.7, previous=previous)
            np.testing.assert_array_equal(labels, fresh_labels)
            self.assertAlmostEqual(energy, fresh_energy)
            previous = labels

    def test_invalid_parameters_are_rejected(self):
        for target, coefficient in [(-1.0, 1.0), (3.0, 1.0),
                                    (1.0, -1.0), (1.0, np.inf), (np.nan, 1.0)]:
            with self.assertRaises(ValueError):
                self.solve((0.0, 0.0), target, coefficient)


if __name__ == '__main__':
    unittest.main()
