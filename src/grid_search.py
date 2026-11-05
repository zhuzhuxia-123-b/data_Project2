from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from math import inf
from time import perf_counter
from typing import Dict, List, Optional, Set, Tuple


Coord = Tuple[int, int]
Grid = List[List[str]]

# 固定邻居生成顺序，使相同优先级下的行为可以复现：
# 上、右、下、左
DIRECTIONS: Tuple[Coord, ...] = (
    (-1, 0),
    (0, 1),
    (1, 0),
    (0, -1),
)


@dataclass
class AStarResult:
    path: List[Coord]
    path_cost: Optional[int]
    solution_depth: Optional[int]
    expanded_nodes: int
    generated_nodes: int
    stale_pops: int
    max_frontier_size: int
    elapsed_seconds: float


def validate_unweighted_grid(
    start: Coord,
    goal: Coord,
    grid: Grid,
) -> None:
    """检查题目一输入是否合法。"""
    if not grid or not grid[0]:
        raise ValueError("grid不能为空")

    width = len(grid[0])
    if any(len(row) != width for row in grid):
        raise ValueError("grid必须是规则的矩形")

    allowed_cells = {"S", "E", ".", "X"}
    for row in grid:
        for cell in row:
            if cell not in allowed_cells:
                raise ValueError(f"题目一出现非法网格内容: {cell!r}")

    for name, coord in (("start", start), ("goal", goal)):
        if (
            not isinstance(coord, tuple)
            or len(coord) != 2
            or not all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in coord
            )
        ):
            raise ValueError(f"{name}必须是包含两个整数的tuple")

        row, col = coord
        if not (0 <= row < len(grid) and 0 <= col < width):
            raise ValueError(f"{name}坐标越界: {coord}")

        if grid[row][col] == "X":
            raise ValueError(f"{name}不能位于障碍物X上")


def heuristic(node: Coord, goal: Coord, mode: str) -> int:
    """返回题目一使用的启发值。"""
    if mode == "zero":
        return 0

    if mode == "manhattan":
        return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

    raise ValueError(
        "heuristic_mode只能是'zero'或'manhattan'"
    )


def reconstruct_path(
    came_from: Dict[Coord, Coord],
    goal: Coord,
) -> List[Coord]:
    """根据前驱关系，从终点反向恢复完整路径。"""
    path = [goal]
    current = goal

    while current in came_from:
        current = came_from[current]
        path.append(current)

    path.reverse()
    return path


def astar_unweighted(
    start: Coord,
    goal: Coord,
    grid: Grid,
    heuristic_mode: str = "manhattan",
) -> AStarResult:
    """
    在单位代价二维网格上运行A*。

    返回路径、代价以及统一定义的搜索统计量。
    无解时path为空，path_cost和solution_depth为None。
    """
    validate_unweighted_grid(start, goal, grid)

    # 确保启发函数名称即使在start == goal时也会被检查。
    heuristic(start, goal, heuristic_mode)

    # 队列项格式：
    # (f值, 稳定顺序编号, 入队时的g值, 坐标)
    open_heap: List[Tuple[int, int, int, Coord]] = []
    tie_counter = count()

    came_from: Dict[Coord, Coord] = {}
    g_score: Dict[Coord, int] = {start: 0}
    closed_set: Set[Coord] = set()

    start_h = heuristic(start, goal, heuristic_mode)
    heappush(
        open_heap,
        (start_h, next(tie_counter), 0, start),
    )

    generated_nodes = 1  # 包含起点
    expanded_nodes = 0
    stale_pops = 0
    max_frontier_size = 1

    started_at = perf_counter()

    while open_heap:
        _, _, queued_g, current = heappop(open_heap)

        # 当前条目不是该节点最新的最佳记录，属于失效队列项。
        if queued_g != g_score.get(current):
            stale_pops += 1
            continue

        # 同一最佳记录不应重复展开。
        if current in closed_set:
            stale_pops += 1
            continue

        closed_set.add(current)
        expanded_nodes += 1

        # A*在目标节点正式出队时结束，而不是在首次生成目标时结束。
        if current == goal:
            elapsed = perf_counter() - started_at
            path = reconstruct_path(came_from, goal)
            cost = g_score[goal]

            return AStarResult(
                path=path,
                path_cost=cost,
                solution_depth=len(path) - 1,
                expanded_nodes=expanded_nodes,
                generated_nodes=generated_nodes,
                stale_pops=stale_pops,
                max_frontier_size=max_frontier_size,
                elapsed_seconds=elapsed,
            )

        current_row, current_col = current

        for row_delta, col_delta in DIRECTIONS:
            neighbor = (
                current_row + row_delta,
                current_col + col_delta,
            )
            neighbor_row, neighbor_col = neighbor

            if not (
                0 <= neighbor_row < len(grid)
                and 0 <= neighbor_col < len(grid[0])
            ):
                continue

            if grid[neighbor_row][neighbor_col] == "X":
                continue

            tentative_g = g_score[current] + 1

            # 只有发现严格更低的代价时才更新前驱并重新入队。
            if tentative_g >= g_score.get(neighbor, inf):
                continue

            came_from[neighbor] = current
            g_score[neighbor] = tentative_g

            # 允许在将来更一般的启发函数下重新打开节点。
            if neighbor in closed_set:
                closed_set.remove(neighbor)

            neighbor_h = heuristic(
                neighbor,
                goal,
                heuristic_mode,
            )
            neighbor_f = tentative_g + neighbor_h

            heappush(
                open_heap,
                (
                    neighbor_f,
                    next(tie_counter),
                    tentative_g,
                    neighbor,
                ),
            )

            generated_nodes += 1
            max_frontier_size = max(
                max_frontier_size,
                len(open_heap),
            )

    elapsed = perf_counter() - started_at

    return AStarResult(
        path=[],
        path_cost=None,
        solution_depth=None,
        expanded_nodes=expanded_nodes,
        generated_nodes=generated_nodes,
        stale_pops=stale_pops,
        max_frontier_size=max_frontier_size,
        elapsed_seconds=elapsed,
    )


def find_shortest_path(
    start: Coord,
    goal: Coord,
    grid: Grid,
) -> List[Coord]:
    """
    课程要求的简洁接口：只返回路径。
    无解时返回空列表。
    """
    return astar_unweighted(
        start,
        goal,
        grid,
        heuristic_mode="manhattan",
    ).path


def path_is_valid(
    path: List[Coord],
    start: Coord,
    goal: Coord,
    grid: Grid,
) -> bool:
    """检查路径首尾、障碍物以及相邻关系。"""
    if not path:
        return False

    if path[0] != start or path[-1] != goal:
        return False

    for row, col in path:
        if not (
            0 <= row < len(grid)
            and 0 <= col < len(grid[0])
        ):
            return False

        if grid[row][col] == "X":
            return False

    for previous, current in zip(path, path[1:]):
        distance = (
            abs(previous[0] - current[0])
            + abs(previous[1] - current[1])
        )
        if distance != 1:
            return False

    return True


def run_given_example() -> None:
    """运行课程给定的题目一样例。"""
    start = (0, 0)
    goal = (3, 4)

    grid = [
        ["S", ".", ".", "X", "."],
        [".", "X", ".", ".", "."],
        [".", "X", ".", "X", "."],
        [".", ".", ".", "X", "E"],
        ["X", "X", ".", ".", "."],
    ]

    result = astar_unweighted(
        start,
        goal,
        grid,
        heuristic_mode="manhattan",
    )

    # 不要求路径必须与PDF逐坐标相同，但必须合法且代价最优。
    assert path_is_valid(result.path, start, goal, grid)
    assert result.path_cost == 7
    assert result.solution_depth == 7

    print("给定样例验证通过")
    print("path:", result.path)
    print("path_cost:", result.path_cost)
    print("solution_depth:", result.solution_depth)
    print("expanded_nodes:", result.expanded_nodes)
    print("generated_nodes:", result.generated_nodes)
    print("stale_pops:", result.stale_pops)
    print("max_frontier_size:", result.max_frontier_size)
    print(
        "elapsed_seconds:",
        f"{result.elapsed_seconds:.9f}",
    )


if __name__ == "__main__":
    run_given_example()
