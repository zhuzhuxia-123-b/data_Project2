"""当代人工智能实验二 · 题目二：加权地形 A*。

使用：与 grid_search.py 放在同一文件夹，运行 python weighted_astar.py。
依赖：仅 Python 标准库和题目一已有的 grid_search.py。

规则：
- 上下左右移动；X 不可通行；进入数字格的代价等于该数字。
- 进入终点 E 的代价为 0；起点 S 的初始代价为 0。
- 如果搜索时重新进入 S，将其代价约定为 1（不影响非负权最优路径）。
- 返回最小总代价；不可达返回 -1。
"""

from heapq import heappop, heappush
from itertools import count
from math import inf
from time import perf_counter
from typing import Dict, List, Optional, Set, Tuple, Union

if __package__:
    from .grid_search import AStarResult, DIRECTIONS, reconstruct_path
else:
    from grid_search import AStarResult, DIRECTIONS, reconstruct_path

Coord = Tuple[int, int]
Cell = Union[int, str]
WeightedGrid = List[List[Cell]]


def entry_cost(cell: Cell) -> Optional[int]:
    """进入一个格子的代价；障碍物 X 返回 None。"""
    if cell == "X":
        return None
    if cell == "E":
        return 0
    if cell == "S":
        return 1
    if isinstance(cell, bool):
        raise ValueError("格子权值不能是布尔值")
    if isinstance(cell, int) and cell >= 0:
        return cell
    if isinstance(cell, str) and cell.isascii() and cell.isdecimal():
        return int(cell)
    raise ValueError(f"不支持的地形格子: {cell!r}；请输入非负整数、S、E或X")


def validate_weighted_grid(start: Coord, goal: Coord, grid: WeightedGrid) -> None:
    if not grid or not grid[0]:
        raise ValueError("grid不能为空")
    width = len(grid[0])
    if any(len(row) != width for row in grid):
        raise ValueError("grid必须是规则矩形")

    for row in grid:
        for cell in row:
            entry_cost(cell)  # 检查整个网格，包括可能不可达的格子

    for name, pos in (("start", start), ("goal", goal)):
        if (
            not isinstance(pos, tuple)
            or len(pos) != 2
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in pos)
        ):
            raise ValueError(f"{name}必须是两个整数构成的tuple")
        r, c = pos
        if not (0 <= r < len(grid) and 0 <= c < width):
            raise ValueError(f"{name}坐标越界: {pos}")
        if grid[r][c] == "X":
            raise ValueError(f"{name}不能位于障碍物X上")

    if start != goal:
        if grid[start[0]][start[1]] != "S":
            raise ValueError("起点坐标处应标记为S")
        if grid[goal[0]][goal[1]] != "E":
            raise ValueError("终点坐标处应标记为E")


def minimum_non_goal_cost(grid: WeightedGrid) -> int:
    """对非终点格，找到统一合法的单步代价下界（允许权值0）。"""
    costs = [
        cost
        for row in grid
        for cell in row
        if cell != "E"
        for cost in [entry_cost(cell)]
        if cost is not None
    ]
    return min(costs, default=0)


def admissible_heuristic(node: Coord, goal: Coord, min_step_cost: int, goal_is_free: bool) -> int:
    """终点进入免费：最多有最后一步免费，所以至少有 max(D-1,0) 步付费。"""
    manhattan = abs(node[0] - goal[0]) + abs(node[1] - goal[1])
    paid_step_lower_bound = max(manhattan - 1, 0) if goal_is_free else manhattan
    return paid_step_lower_bound * min_step_cost


def astar_weighted(
    start: Coord,
    goal: Coord,
    grid: WeightedGrid,
    heuristic_mode: str = "adjusted_manhattan",
) -> AStarResult:
    """返回带统计指标的 A* 结果，方便后续做对照实验。

    heuristic_mode:
      zero                -> h=0，最优（Dijkstra）
      manhattan           -> 普通曼哈顿距离，不乘权值，E免费时不保证最优
      adjusted_manhattan  -> 考虑 E 免费进入的可采纳启发函数
    """
    validate_weighted_grid(start, goal, grid)
    if heuristic_mode not in ("zero", "manhattan", "adjusted_manhattan"):
        raise ValueError("heuristic_mode只能是 'zero'、'manhattan' 或 'adjusted_manhattan'")

    min_cost = minimum_non_goal_cost(grid)
    goal_is_free = grid[goal[0]][goal[1]] == "E"

    def h(node: Coord) -> int:
        if heuristic_mode == "zero":
            return 0
        if heuristic_mode == "manhattan":
            # 教学反例模式：若 E 免费进入或存在低于1的权值，可能高估而失去最优保证。
            return abs(node[0] - goal[0]) + abs(node[1] - goal[1])
        return admissible_heuristic(node, goal, min_cost, goal_is_free)

    # (f, 稳定入队顺序, 入队时的g, 坐标)：保证同一输入的运行结果可复现。
    open_heap: List[Tuple[int, int, int, Coord]] = []
    tie_counter = count()
    heappush(open_heap, (h(start), next(tie_counter), 0, start))

    g_score: Dict[Coord, int] = {start: 0}
    came_from: Dict[Coord, Coord] = {}
    closed_set: Set[Coord] = set()
    expanded_nodes = 0
    generated_nodes = 1
    stale_pops = 0
    max_frontier_size = 1
    t0 = perf_counter()

    while open_heap:
        _, _, queued_g, current = heappop(open_heap)
        if queued_g != g_score.get(current) or current in closed_set:
            stale_pops += 1
            continue

        closed_set.add(current)
        expanded_nodes += 1

        if current == goal:
            path = reconstruct_path(came_from, goal)
            return AStarResult(
                path=path,
                path_cost=g_score[goal],
                solution_depth=len(path) - 1,
                expanded_nodes=expanded_nodes,
                generated_nodes=generated_nodes,
                stale_pops=stale_pops,
                max_frontier_size=max_frontier_size,
                elapsed_seconds=perf_counter() - t0,
            )

        r, c = current
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < len(grid) and 0 <= nc < len(grid[0])):
                continue
            move_cost = entry_cost(grid[nr][nc])
            if move_cost is None:
                continue

            neighbor = (nr, nc)
            tentative_g = g_score[current] + move_cost
            if tentative_g >= g_score.get(neighbor, inf):
                continue

            g_score[neighbor] = tentative_g
            came_from[neighbor] = current
            # 同题目一：只有发现更优路径，才允许重新打开已展开节点。
            closed_set.discard(neighbor)
            heappush(
                open_heap,
                (tentative_g + h(neighbor), next(tie_counter), tentative_g, neighbor),
            )
            generated_nodes += 1
            max_frontier_size = max(max_frontier_size, len(open_heap))

    return AStarResult(
        path=[],
        path_cost=None,
        solution_depth=None,
        expanded_nodes=expanded_nodes,
        generated_nodes=generated_nodes,
        stale_pops=stale_pops,
        max_frontier_size=max_frontier_size,
        elapsed_seconds=perf_counter() - t0,
    )


def find_min_cost(start: Coord, goal: Coord, grid: WeightedGrid) -> int:
    """符合课程要求的简洁接口：最小移动代价，若无解则 -1。"""
    cost = astar_weighted(start, goal, grid).path_cost
    return -1 if cost is None else cost


def sum_path_cost(path: List[Coord], grid: WeightedGrid) -> int:
    """单独根据返回路径累加进入格子的代价，用来检查A*的路径代价。"""
    total = 0
    for r, c in path[1:]:
        cost = entry_cost(grid[r][c])
        if cost is None:
            raise ValueError("路径不能经过障碍物")
        total += cost
    return total


def run_given_example() -> None:
    start, goal = (0, 0), (4, 4)
    grid: WeightedGrid = [
        ["S", 1, 1, 8, 1],
        [1, 8, 3, 1, 1],
        [1, 8, 1, 8, 1],
        [1, 1, 1, 8, 1],
        [8, 8, 1, 1, "E"],
    ]
    result = astar_weighted(start, goal, grid)
    baseline = astar_weighted(start, goal, grid, heuristic_mode="zero")

    assert result.path[0] == start and result.path[-1] == goal
    assert all(
        abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
        for a, b in zip(result.path, result.path[1:])
    )
    assert result.path_cost == 7, f"与PDF预期不符，得到{result.path_cost}"
    assert sum_path_cost(result.path, grid) == 7, "路径累计代价检查失败"
    assert baseline.path_cost == result.path_cost, "Dijkstra与A*最优代价不一致"
    assert find_min_cost(start, goal, grid) == 7

    print("=== 实验二 / 题目二：加权地形 A* ===")
    print("[PASS] PDF给定样例最小总代价 = 7")
    print("path:", result.path)
    print("path_cost:", result.path_cost)
    print("solution_depth:", result.solution_depth)
    print("进入各格代价:", [entry_cost(grid[r][c]) for r, c in result.path[1:]])
    print("累计路径代价校验:", sum_path_cost(result.path, grid))
    print("A* expanded_nodes:", result.expanded_nodes)
    print("h=0 expanded_nodes:", baseline.expanded_nodes)

    blocked = [["S", "X", "E"], ["X", "X", "X"]]
    assert find_min_cost((0, 0), (0, 2), blocked) == -1
    print("[PASS] 无解地图返回 -1")

    adjacent = [["S", "E"]]
    assert find_min_cost((0, 0), (0, 1), adjacent) == 0
    print("[PASS] 相邻终点 E 免费进入，总代价 = 0")

    one_cell = [["S"]]
    assert find_min_cost((0, 0), (0, 0), one_cell) == 0
    print("[PASS] 起终点重合，总代价 = 0")
    print("=== 题目二：基础功能检查全部通过 ===")


if __name__ == "__main__":
    run_given_example()
