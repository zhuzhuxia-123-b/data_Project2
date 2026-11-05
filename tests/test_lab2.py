from __future__ import annotations

import math
import importlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from grid_search import astar_unweighted, find_shortest_path, path_is_valid
from experiment_q1 import reference_bfs_cost
from weighted_astar import astar_weighted, find_min_cost, sum_path_cost
from game_tree import (
    TreeNode,
    alpha_beta_search as tree_ab,
    build_pdf_example,
    minimax_search as tree_minimax,
)
from treasure_game import initial_state, take_treasure, minimax_search, alpha_beta_search
from experiment_innovation import alpha_beta_gain_ordered
from treasure_distance import NAMES, build_distance_matrix


class TestUnweightedAStar(unittest.TestCase):
    def check_optimal(self, grid, start, goal):
        result = astar_unweighted(start, goal, grid)
        baseline = reference_bfs_cost(start, goal, grid)
        self.assertEqual(result.path_cost, baseline)
        if baseline is None:
            self.assertEqual(result.path, [])
            self.assertEqual(find_shortest_path(start, goal, grid), [])
        else:
            self.assertTrue(path_is_valid(result.path, start, goal, grid))
            self.assertEqual(len(result.path) - 1, baseline)
        return result

    def test_pdf_sample(self):
        grid = [list('S..X.'), list('.X...'), list('.X.X.'), list('...XE'), list('XX...')]
        self.assertEqual(self.check_optimal(grid, (0,0), (3,4)).path_cost, 7)

    def test_start_equals_goal(self):
        r = self.check_optimal([['S']], (0,0), (0,0))
        self.assertEqual(r.path, [(0,0)])

    def test_no_path(self):
        self.check_optimal([['S','X','E'],['X','X','X']], (0,0), (0,2))

    def test_single_corridor(self):
        grid = [list('S....'),list('XXXX.'),list('E....')]
        self.check_optimal(grid, (0,0), (2,0))

    def test_equal_length_shortest_paths(self):
        self.check_optimal([['S','.','.'],['.','.','.'],['.','.','E']], (0,0), (2,2))

    def test_start_goal_at_edges(self):
        self.check_optimal([list('S...E')], (0,0), (0,4))

    def test_negative_coordinates(self):
        with self.assertRaises(ValueError):
            astar_unweighted((-1,0),(0,1),[list('SE')])

    def test_out_of_bounds(self):
        with self.assertRaises(ValueError):
            astar_unweighted((0,0),(0,2),[list('SE')])

    def test_jagged_grid(self):
        with self.assertRaises(ValueError):
            astar_unweighted((0,0),(1,0),[['S','E'],['.']])

    def test_empty_grid(self):
        with self.assertRaises(ValueError):
            astar_unweighted((0,0),(0,0),[])

    def test_start_obstacle(self):
        with self.assertRaises(ValueError):
            astar_unweighted((0,0),(0,1),[list('XE')])

    def test_goal_obstacle(self):
        with self.assertRaises(ValueError):
            astar_unweighted((0,0),(0,1),[list('SX')])

    def test_illegal_grid_character(self):
        with self.assertRaises(ValueError):
            astar_unweighted((0,0),(0,2),[list('S?E')])

    def test_illegal_heuristic(self):
        with self.assertRaises(ValueError):
            astar_unweighted((0,0),(0,1),[list('SE')],heuristic_mode='invalid')

    def test_boolean_coordinate(self):
        with self.assertRaises(ValueError):
            astar_unweighted((True, 0), (0, 1), [list('SE')])


class TestWeightedAStar(unittest.TestCase):
    def test_pdf_sample_and_path_cost(self):
        grid = [
            ['S', 1, 1, 8, 1],
            [1, 8, 3, 1, 1],
            [1, 8, 1, 8, 1],
            [1, 1, 1, 8, 1],
            [8, 8, 1, 1, 'E'],
        ]
        result = astar_weighted((0, 0), (4, 4), grid)
        self.assertEqual(result.path_cost, 7)
        self.assertEqual(sum_path_cost(result.path, grid), 7)
        self.assertEqual(find_min_cost((0, 0), (4, 4), grid), 7)

    def test_zero_cost_and_unreachable_cases(self):
        self.assertEqual(find_min_cost((0, 0), (0, 1), [['S', 'E']]), 0)
        self.assertEqual(find_min_cost((0, 0), (0, 0), [['S']]), 0)
        self.assertEqual(
            find_min_cost((0, 0), (0, 2), [['S', 'X', 'E']]),
            -1,
        )

    def test_invalid_weighted_grids(self):
        invalid_cases = (
            ((0, 0), (1, 0), [['S', 'E'], [1]]),
            ((0, 0), (0, 2), [['S', -1, 'E']]),
            ((0, 0), (0, 2), [['S', True, 'E']]),
            ((0, 0), (0, 1), [['X', 'E']]),
            ((0, 0), (0, 1), [['S', 'X']]),
        )
        for start, goal, grid in invalid_cases:
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                astar_weighted(start, goal, grid)


class TestGameTree(unittest.TestCase):
    def test_pdf_sample_pruning_and_order_invariance(self):
        for order in ('left_to_right', 'right_to_left'):
            root = build_pdf_example()
            mm = tree_minimax(root, order=order)
            ab = tree_ab(root, order=order)
            self.assertEqual(mm.best_value, 5)
            self.assertEqual(ab.best_value, 5)
            self.assertLessEqual(ab.visited_nodes, mm.visited_nodes)
        left = tree_ab(build_pdf_example(), order='left_to_right')
        self.assertGreater(left.cutoff_events, 0)
        self.assertLess(left.visited_nodes, left.total_nodes)

    def test_invalid_leaf_and_cycle(self):
        with self.assertRaises(ValueError):
            tree_minimax(TreeNode())
        cyclic = TreeNode(children=[])
        cyclic.children.append(cyclic)
        with self.assertRaises(ValueError):
            tree_ab(cyclic)

    def test_min_root(self):
        root = TreeNode(children=[TreeNode(value=3), TreeNode(value=-2)])
        self.assertEqual(tree_minimax(root, root_is_max=False).best_value, -2)
        self.assertEqual(tree_ab(root, root_is_max=False).best_value, -2)


class TestPackageImports(unittest.TestCase):
    def test_src_modules_can_be_imported_as_package(self):
        for module_name in (
            'src.weighted_astar',
            'src.treasure_distance',
            'src.treasure_game',
            'src.experiment_q1',
            'src.experiment_q2',
            'src.experiment_q3',
            'src.experiment_q4',
            'src.experiment_innovation',
            'src.experiment_precomputation',
        ):
            with self.subTest(module=module_name):
                self.assertIsNotNone(importlib.import_module(module_name))


class TestExtensions(unittest.TestCase):
    def test_ordinary_manhattan_counterexample(self):
        grid = [['S',1,1,1,'E'],[0,0,0,0,0]]
        self.assertEqual(astar_weighted((0,0),(0,4),grid,'zero').path_cost, 0)
        self.assertEqual(astar_weighted((0,0),(0,4),grid,'adjusted_manhattan').path_cost, 0)
        self.assertEqual(astar_weighted((0,0),(0,4),grid,'manhattan').path_cost, 3)

    def test_tree_single_leaf_depth_zero(self):
        for search in (tree_minimax, tree_ab):
            result = search(TreeNode(value=7))
            self.assertEqual(result.best_value,7)
            self.assertEqual(result.max_depth_reached, 0)

    def test_tree_pdf_depth(self):
        from game_tree import build_pdf_example
        for search in (tree_minimax, tree_ab):
            result = search(build_pdf_example())
            self.assertEqual(result.max_depth_reached, 3)
            self.assertEqual(result.best_value, 5)

    def test_game_depth_zero(self):
        distances, _ = build_distance_matrix()
        state = initial_state()
        for search in (minimax_search, alpha_beta_search, alpha_beta_gain_ordered):
            result = search(state, distances, 0)
            self.assertEqual(result.max_depth_reached,0)
            self.assertIsNone(result.best_target)
            self.assertEqual(result.best_value,state.utility)

    def test_game_depth_two_four(self):
        distances, _ = build_distance_matrix()
        state = initial_state()
        for depth, action, utility in ((2,'T2',-1),(4,'T3',3)):
            for search in (minimax_search, alpha_beta_search, alpha_beta_gain_ordered):
                result = search(state, distances, depth)
                self.assertEqual(result.max_depth_reached,depth)
                self.assertEqual((result.best_target,result.best_value),(action,utility))

    def test_reject_unreachable_treasure(self):
        distances, _ = build_distance_matrix()
        state = initial_state()
        for bad in (-1, math.inf, -math.inf, float('nan')):
            broken = {name: dict(row) for name,row in distances.items()}
            broken['A']['T2'] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                take_treasure(state,'T2',broken)
        broken = {name: dict(row) for name,row in distances.items()}
        del broken['A']['T2']
        with self.assertRaises(ValueError):
            take_treasure(state,'T2',broken)

    def test_complete_distance_matrix(self):
        distances, _ = build_distance_matrix()
        expected = {
            'A':  [0, 12, 10, 4, 8, 8],
            'B':  [12, 0, 8, 8, 4, 10],
            'T1': [10, 8, 0, 6, 6, 12],
            'T2': [4, 8, 6, 0, 6, 6],
            'T3': [8, 4, 6, 6, 0, 6],
            'T4': [8, 10, 12, 6, 6, 0],
        }
        for source in NAMES:
            self.assertEqual(
                [distances[source][target] for target in NAMES],
                expected[source],
            )
            self.assertEqual(distances[source][source], 0)
            for target in NAMES:
                self.assertEqual(distances[source][target], distances[target][source])

    def test_treasure_state_transition_and_duplicate_rejection(self):
        distances, _ = build_distance_matrix()
        state = initial_state()
        after, gain = take_treasure(state, 'T3', distances)
        self.assertEqual(gain, -1)
        self.assertEqual(after.pos_a, 'T3')
        self.assertEqual(after.pos_b, 'B')
        self.assertEqual(after.remaining, ('T1', 'T2', 'T4'))
        self.assertEqual((after.score_a, after.score_b), (-1, 0))
        self.assertEqual(after.current_player, 'B')
        with self.assertRaises(ValueError):
            take_treasure(after, 'T3', distances)

    def test_alpha_beta_reduces_game_search(self):
        distances, _ = build_distance_matrix()
        state = initial_state()
        mm = minimax_search(state, distances, 4)
        ab = alpha_beta_search(state, distances, 4)
        self.assertEqual((mm.best_target, mm.best_value), ('T3', 3))
        self.assertEqual((ab.best_target, ab.best_value), ('T3', 3))
        self.assertLess(ab.visited_states, mm.visited_states)
        self.assertLess(ab.evaluated_leaves, mm.evaluated_leaves)


if __name__ == '__main__':
    unittest.main()
