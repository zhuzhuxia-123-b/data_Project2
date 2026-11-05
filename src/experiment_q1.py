"""实验二 · 题目一：A* 正确性检查与启发函数对照实验。

使用方式：将本文件与 grid_search.py 放在同一目录，运行 python experiment_q1.py。
只依赖 Python 标准库；不会修改 grid_search.py。
"""

from collections import deque
from csv import DictWriter
from pathlib import Path
from statistics import median
from typing import List, Optional, Tuple

if __package__:
    from .grid_search import astar_unweighted, find_shortest_path, path_is_valid
else:
    from grid_search import astar_unweighted, find_shortest_path, path_is_valid

Coord = Tuple[int, int]
Grid = List[List[str]]
REPEATS = 200  # 运行时间很短，重复多次后取中位数；不用于判断路径正确性
OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def reference_bfs_cost(start: Coord, goal: Coord, grid: Grid) -> Optional[int]:
    """独立 BFS 正确性基准：单位边权时的最短距离。None 代表无路径。"""
    q = deque([(start, 0)])
    visited = {start}
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    while q:
        (r, c), steps = q.popleft()
        if (r, c) == goal:
            return steps
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < len(grid)
                and 0 <= nc < len(grid[0])
                and grid[nr][nc] != "X"
                and (nr, nc) not in visited
            ):
                visited.add((nr, nc))
                q.append(((nr, nc), steps + 1))
    return None


def open_grid(size: int, start: Coord, goal: Coord) -> Grid:
    grid = [["." for _ in range(size)] for _ in range(size)]
    grid[start[0]][start[1]] = "S"
    grid[goal[0]][goal[1]] = "E"
    return grid


def run_case(name: str, start: Coord, goal: Coord, grid: Grid) -> List[dict]:
    true_cost = reference_bfs_cost(start, goal, grid)
    rows = []
    for mode in ("zero", "manhattan"):
        result = astar_unweighted(start, goal, grid, heuristic_mode=mode)
        # 不能仅检查是否找到路径，还要独立验证它是最短路径。
        assert result.path_cost == true_cost, (name, mode, result.path_cost, true_cost)
        if true_cost is None:
            assert result.path == [] and result.solution_depth is None
        else:
            assert path_is_valid(result.path, start, goal, grid), (name, mode)
            assert len(result.path) - 1 == true_cost

        measurements = [result.elapsed_seconds]
        for _ in range(REPEATS - 1):
            followup = astar_unweighted(start, goal, grid, heuristic_mode=mode)
            assert followup.path_cost == true_cost
            measurements.append(followup.elapsed_seconds)

        row = {
            "case": name,
            "heuristic": mode,
            "optimal_cost": "unreachable" if true_cost is None else true_cost,
            "expanded_nodes": result.expanded_nodes,
            "generated_nodes": result.generated_nodes,
            "stale_pops": result.stale_pops,
            "max_frontier_size": result.max_frontier_size,
            "median_time_ms": round(median(measurements) * 1000, 6),
            "repeats": REPEATS,
        }
        rows.append(row)

    zero, manhattan = rows
    saving = (zero["expanded_nodes"] - manhattan["expanded_nodes"]) / max(1, zero["expanded_nodes"]) * 100
    print(f"\n[{name}] BFS基准最短代价={true_cost}")
    for row in rows:
        print(
            f"  {row['heuristic']:9s}  "
            f"展开={row['expanded_nodes']:3d}  生成={row['generated_nodes']:3d}  "
            f"中位耗时={row['median_time_ms']:.4f}ms"
        )
    print(f"  曼哈顿距离减少展开节点：{saving:.1f}%")
    return rows


def check_edge_cases() -> None:
    print("\n=== 边界与异常场景测试 ===")
    no_path_grid = [["S", "X", "E"], ["X", "X", "X"], [".", ".", "."]]
    for mode in ("zero", "manhattan"):
        result = astar_unweighted((0, 0), (0, 2), no_path_grid, mode)
        assert result.path == [] and result.path_cost is None
    assert find_shortest_path((0, 0), (0, 2), no_path_grid) == []
    print("[PASS] 无解地图：返回空列表 []")

    one_cell = [["S"]]
    for mode in ("zero", "manhattan"):
        result = astar_unweighted((0, 0), (0, 0), one_cell, mode)
        assert result.path == [(0, 0)]
        assert result.path_cost == 0 and result.solution_depth == 0
    print("[PASS] 起点与终点重合：路径只有一个坐标，代价为0")

    try:
        find_shortest_path((0, 0), (0, 2), [["S", "."], [".", "E"]])
    except ValueError:
        print("[PASS] 越界坐标：正确抛出 ValueError")
    else:
        raise AssertionError("越界坐标应当抛出 ValueError")


def main() -> None:
    print("=== 实验二 / 题目一：A* 对照实验 ===")
    print(f"说明：耗时统计为 {REPEATS} 次运行的中位数；单位ms。")
    given_grid = [
        ["S", ".", ".", "X", "."],
        [".", "X", ".", ".", "."],
        [".", "X", ".", "X", "."],
        [".", ".", ".", "X", "E"],
        ["X", "X", ".", ".", "."],
    ]
    given_path = find_shortest_path((0, 0), (3, 4), given_grid)
    assert len(given_path) - 1 == 7
    print(f"\nPDF给定样例路径：{given_path}")

    cases = [
        ("PDF_5x5_maze", (0, 0), (3, 4), given_grid),
        ("OPEN_15x15_same_row", (0, 0), (0, 14), open_grid(15, (0, 0), (0, 14))),
        ("OPEN_15x15_opposite_corners", (0, 0), (14, 14), open_grid(15, (0, 0), (14, 14))),
    ]
    records = []
    for name, start, goal, grid in cases:
        records.extend(run_case(name, start, goal, grid))

    check_edge_cases()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "q1_astar_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    print("\n=== 本次检查全部通过 ===")
    print(f"对照实验结果已保存：{csv_path}")
    print("注意：毫秒级耗时易受机器负载影响；讨论搜索效率时，应优先结合节点展开数。")


if __name__ == "__main__":
    main()
