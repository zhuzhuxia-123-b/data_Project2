"""当代人工智能 · 实验二 / 题目二：加权 A* 对照实验与独立验证。

使用：将本文件放在 grid_search.py、weighted_astar.py 同一目录，
      运行 python experiment_q2.py。
依赖：仅 Python 标准库 + 已完成的 weighted_astar.py。
输出：results/q2_weighted_astar_comparison.csv。

对照：h=0、ordinary manhattan（不保证最优）、adjusted_manhattan（保证最优）。
验证：独立 Dijkstra、路径代价累加、最少步数与最小代价区别、
      随机地图交叉验证、启发函数可采纳性检查。
"""

from collections import deque
from csv import DictWriter
from heapq import heappop, heappush
from math import inf
from pathlib import Path
from random import Random
from statistics import median
from typing import List, Optional, Tuple

if __package__:
    from .weighted_astar import (
        WeightedGrid,
        admissible_heuristic,
        astar_weighted,
        find_min_cost,
        minimum_non_goal_cost,
    )
else:
    from weighted_astar import (
        WeightedGrid,
        admissible_heuristic,
        astar_weighted,
        find_min_cost,
        minimum_non_goal_cost,
    )

Coord = Tuple[int, int]
REPEATS = 200
RANDOM_TESTS = 200
RANDOM_SEED = 20261031
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def independent_entry_cost(cell: object) -> Optional[int]:
    """独立于被测实现的地形规则，用于构造参考求解器。"""
    if cell == "X":
        return None
    if cell == "E":
        return 0
    if cell == "S":
        return 1
    if isinstance(cell, bool):
        raise ValueError("不支持布尔权值")
    if isinstance(cell, int) and cell >= 0:
        return cell
    if isinstance(cell, str) and cell.isascii() and cell.isdecimal():
        return int(cell)
    raise ValueError(f"非法地形格子: {cell!r}")


def neighbors(pos: Coord, grid: WeightedGrid):
    r, c = pos
    rows, cols = len(grid), len(grid[0])
    for dr, dc in DIRECTIONS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < rows and 0 <= cc < cols and grid[rr][cc] != "X":
            yield (rr, cc)


def reference_dijkstra_cost(start: Coord, goal: Coord, grid: WeightedGrid) -> Optional[int]:
    """独立 Dijkstra 基准；无解返回 None。"""
    pq = [(0, start)]
    dist = {start: 0}
    while pq:
        g, pos = heappop(pq)
        if g != dist[pos]:
            continue
        if pos == goal:
            return g
        for nxt in neighbors(pos, grid):
            price = independent_entry_cost(grid[nxt[0]][nxt[1]])
            assert price is not None
            alt = g + price
            if alt < dist.get(nxt, inf):
                dist[nxt] = alt
                heappush(pq, (alt, nxt))
    return None


def reference_reverse_distances(goal: Coord, grid: WeightedGrid):
    """反向 Dijkstra 求每格到 E 的真实最小代价，用于验证 h <= h*。

    原方向 p -> q 的费用是“进入 q”的权值，所以反向从 q 到 p
    的费用仍是进入原方向 q（即当前点）的权值。
    """
    distances = {goal: 0}
    pq = [(0, goal)]
    while pq:
        d, current = heappop(pq)
        if d != distances[current]:
            continue
        reverse_edge_cost = independent_entry_cost(grid[current[0]][current[1]])
        assert reverse_edge_cost is not None
        for predecessor in neighbors(current, grid):
            nd = d + reverse_edge_cost
            if nd < distances.get(predecessor, inf):
                distances[predecessor] = nd
                heappush(pq, (nd, predecessor))
    return distances


def independent_path_cost(path: List[Coord], grid: WeightedGrid) -> int:
    return sum(independent_entry_cost(grid[r][c]) for r, c in path[1:])


def check_path(path: List[Coord], start: Coord, goal: Coord, grid: WeightedGrid, cost: Optional[int]):
    if cost is None:
        assert path == [], "无解时路径应为空"
        return
    assert path and path[0] == start and path[-1] == goal, "起终点不正确"
    for r, c in path:
        assert 0 <= r < len(grid) and 0 <= c < len(grid[0])
        assert grid[r][c] != "X", "路径穿过了障碍物"
    for a, b in zip(path, path[1:]):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1, "只能上下左右移动"
    assert independent_path_cost(path, grid) == cost, "返回代价与路径累计代价不一致"


def shortest_steps_path(start: Coord, goal: Coord, grid: WeightedGrid) -> List[Coord]:
    """忽略地形权值的 BFS，仅用于展示最少步数不等于最小代价。"""
    q = deque([start])
    parent = {start: None}
    while q:
        pos = q.popleft()
        if pos == goal:
            path = []
            while pos is not None:
                path.append(pos)
                pos = parent[pos]
            return path[::-1]
        for nxt in neighbors(pos, grid):
            if nxt not in parent:
                parent[nxt] = pos
                q.append(nxt)
    return []


def run_case(name: str, start: Coord, goal: Coord, grid: WeightedGrid):
    """对三种启发函数做真实统计；不对普通曼哈顿强制最优断言。"""
    truth = reference_dijkstra_cost(start, goal, grid)
    rows = []
    for mode in ("zero", "manhattan", "adjusted_manhattan"):
        guaranteed = int(mode != "manhattan")
        result = astar_weighted(start, goal, grid, mode)
        if guaranteed:
            assert result.path_cost == truth, (name, mode, result.path_cost, truth)
        # 即使算法返回次优路径，仍必须验证路径合法、累计代价真实。
        check_path(result.path, start, goal, grid, result.path_cost)
        assert result.solution_depth == (None if result.path_cost is None else len(result.path)-1)
        measurements = [result.elapsed_seconds]
        signature = (result.path_cost, result.expanded_nodes, result.generated_nodes,
                     result.stale_pops, result.max_frontier_size)
        for _ in range(REPEATS - 1):
            followup = astar_weighted(start, goal, grid, mode)
            assert (followup.path_cost, followup.expanded_nodes, followup.generated_nodes,
                    followup.stale_pops, followup.max_frontier_size) == signature
            measurements.append(followup.elapsed_seconds)
        same = result.path_cost == truth
        rows.append({
            "case": name,
            "heuristic": mode,
            "reference_optimal_cost": "unreachable" if truth is None else truth,
            "returned_cost": "unreachable" if result.path_cost is None else result.path_cost,
            "is_optimal": int(same),
            "optimality_gap": "" if truth is None else result.path_cost - truth,
            "optimality_guaranteed": guaranteed,
            "solution_depth": "" if result.solution_depth is None else result.solution_depth,
            "expanded_nodes": result.expanded_nodes,
            "generated_nodes": result.generated_nodes,
            "stale_pops": result.stale_pops,
            "max_frontier_size": result.max_frontier_size,
            "median_time_ms": round(median(measurements) * 1000, 6),
            "repeats": REPEATS,
        })

    zero, ordinary, adjusted = rows
    denom = zero["expanded_nodes"]
    saved = 100 * (denom - adjusted["expanded_nodes"]) / denom if denom else 0.0
    print(f"\n[{name}] 独立Dijkstra最小代价={truth}")
    for item in rows:
        print(f"  {item['heuristic']:19s} 返回={item['returned_cost']!s:>11s} "
              f"最优={item['is_optimal']} 展开={item['expanded_nodes']:3d} "
              f"生成={item['generated_nodes']:3d} 中位耗时={item['median_time_ms']:.4f}ms")
    print(f"  修正曼哈顿减少展开节点：{saved:.1f}%")
    return rows


def mixed_grid(size: int = 12) -> WeightedGrid:
    """固定种子混合地形：有障碍、平原、沼泽、高代价地形，保留一条可行走廊。"""
    rng = Random(2026)
    grid: WeightedGrid = []
    for _ in range(size):
        row = ["X" if rng.random() < 0.17 else rng.choice([1, 1, 1, 2, 3, 8])
               for _ in range(size)]
        grid.append(row)
    for col in range(size):
        grid[0][col] = 1
    for row in range(size):
        grid[row][size - 1] = 1
    grid[0][0] = "S"
    grid[-1][-1] = "E"
    return grid


def random_cross_checks():
    rng = Random(RANDOM_SEED)
    ordinary_matches = ordinary_counterexamples = 0
    for i in range(RANDOM_TESTS):
        rows, cols = rng.randint(3, 8), rng.randint(3, 8)
        grid: WeightedGrid = [
            ["X" if rng.random() < 0.21 else rng.choice([0, 1, 1, 1, 3, 8])
             for _ in range(cols)] for _ in range(rows)
        ]
        start, goal = (0, 0), (rows - 1, cols - 1)
        grid[0][0] = "S"
        grid[-1][-1] = "E"
        truth = reference_dijkstra_cost(start, goal, grid)
        for mode in ("zero", "manhattan", "adjusted_manhattan"):
            result = astar_weighted(start, goal, grid, mode)
            if mode != "manhattan":
                assert result.path_cost == truth, (i, mode, truth, result.path_cost)
            else:
                ordinary_matches += int(result.path_cost == truth)
                ordinary_counterexamples += int(result.path_cost != truth)
            check_path(result.path, start, goal, grid, result.path_cost)

        min_cost = minimum_non_goal_cost(grid)
        true_distance_to_e = reference_reverse_distances(goal, grid)
        for pos, exact_remaining_cost in true_distance_to_e.items():
            h = admissible_heuristic(pos, goal, min_cost, True)
            assert 0 <= h <= exact_remaining_cost, (
                f"地图{i}处{pos}: h={h}, 真正剩余代价={exact_remaining_cost}"
            )
    print(f"[PASS] {RANDOM_TESTS}组随机加权地图：zero/adjusted始终与独立Dijkstra一致")
    print(f"[记录] ordinary Manhattan 随机一致率={ordinary_matches}/{RANDOM_TESTS} "
          f"({ordinary_matches/RANDOM_TESTS:.1%})；发现次优/不一致={ordinary_counterexamples}组")
    print("[PASS] 修正启发函数对每个可到达终点格满足 h(n) <= 真正剩余代价")
    return ordinary_matches, ordinary_counterexamples


def main():
    print("=== 实验二 / 题目二：加权 A* 性能与正确性对照 ===")
    print(f"每个配置运行{REPEATS}次，耗时报告中位数（ms）。")

    pdf_map: WeightedGrid = [
        ["S", 1, 1, 8, 1],
        [1, 8, 3, 1, 1],
        [1, 8, 1, 8, 1],
        [1, 1, 1, 8, 1],
        [8, 8, 1, 1, "E"],
    ]
    shorter_steps_costlier: WeightedGrid = [
        ["S", 9, 9, 9, "E"],
        [1, 1, 1, 1, 1],
    ]
    open_15x15: WeightedGrid = [[1] * 15 for _ in range(15)]
    open_15x15[0][0] = "S"
    open_15x15[0][14] = "E"
    zero_cost_map: WeightedGrid = [
        ["S", 0, 1, 1, 1],
        [1, 8, 8, 8, 1],
        [1, 1, 1, 1, 1],
        [1, 8, 8, 8, 1],
        [1, 1, 1, 1, "E"],
    ]
    # 可重复的反例：直接走顶行代价3，绕行底部0权值通道最优代价0。
    # 当前稳定队列顺序下，普通Manhattan高估了绕行方向，提前出队E并返回3。
    nonoptimal_counterexample: WeightedGrid = [
        ["S", 1, 1, 1, "E"],
        [0, 0, 0, 0, 0],
    ]
    cases = [
        ("PDF_5x5_weighted", (0, 0), (4, 4), pdf_map),
        ("FEWER_STEPS_HIGHER_COST", (0, 0), (0, 4), shorter_steps_costlier),
        ("OPEN_15x15_same_row", (0, 0), (0, 14), open_15x15),
        ("MIXED_12x12_seed_2026", (0, 0), (11, 11), mixed_grid()),
        ("ZERO_WEIGHT_5x5", (0, 0), (4, 4), zero_cost_map),
        ("ORDINARY_MANHATTAN_COUNTEREXAMPLE", (0, 0), (0, 4), nonoptimal_counterexample),
    ]

    records = []
    for name, start, goal, grid in cases:
        records.extend(run_case(name, start, goal, grid))

    counter = {r["heuristic"]: r for r in records if r["case"] == "ORDINARY_MANHATTAN_COUNTEREXAMPLE"}
    assert counter["zero"]["returned_cost"] == counter["adjusted_manhattan"]["returned_cost"] == 0
    assert counter["manhattan"]["returned_cost"] == 3
    assert counter["manhattan"]["is_optimal"] == 0 and counter["manhattan"]["optimality_gap"] == 3
    print("[PASS] 确定性反例：独立Dijkstra=0、ordinary Manhattan=3（次优）")

    print("\n=== 最少步数与最小移动代价的区别 ===")
    steps_path = shortest_steps_path((0, 0), (0, 4), shorter_steps_costlier)
    cost_path = astar_weighted((0, 0), (0, 4), shorter_steps_costlier).path
    print(f"BFS最少步数路径：{len(steps_path) - 1}步，代价={independent_path_cost(steps_path, shorter_steps_costlier)}")
    print(f"A*最小代价路径：{len(cost_path) - 1}步，代价={independent_path_cost(cost_path, shorter_steps_costlier)}")
    assert len(steps_path) - 1 == 4 and independent_path_cost(steps_path, shorter_steps_costlier) == 27
    assert len(cost_path) - 1 == 6 and independent_path_cost(cost_path, shorter_steps_costlier) == 5
    print("[PASS] 最少步数不等于最小代价：4步/代价27 vs 6步/代价5")

    print("\n=== 随机交叉验证与启发函数检查 ===")
    random_cross_checks()
    assert find_min_cost((0, 0), (4, 4), pdf_map) == 7
    assert find_min_cost((0, 0), (0, 1), [["S", "E"]]) == 0
    assert find_min_cost((0, 0), (0, 0), [["S"]]) == 0
    assert find_min_cost((0, 0), (0, 2), [["S", "X", "E"]]) == -1
    print("[PASS] PDF样例、紧邻免费终点、起终点重合、无解地图")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "q2_weighted_astar_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    print("\n=== 题目二：对照实验全部通过 ===")
    print("实验数据已保存：", csv_path)
    print("说明：微小的毫秒级时间差可能受机器负载影响；优先比较搜索节点数。")


if __name__ == "__main__":
    main()
