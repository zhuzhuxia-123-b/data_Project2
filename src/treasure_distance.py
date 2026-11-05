"""当代人工智能实验二 · 题目四（第1阶段）：宝藏地图的 6×6 A* 距离矩阵。

依赖：同目录中的 grid_search.py、weighted_astar.py（仅Python标准库）。
运行：python treasure_distance.py

特别注意：题目二进入 E 的代价为0，而题目四进入 A、B、T 的代价均为1。
为了正确复用题目二的 A*，对每一对起终点生成临时网格：起点标 S，
终点标 E，其余特殊格按1处理。A*在临时网格上算出的路径代价需加1，
才是题目四的真实移动代价（起终点不同的情况下）。
"""

import csv
from heapq import heappop, heappush
from math import inf
from pathlib import Path
from typing import Dict, List, Tuple

if __package__:
    from .weighted_astar import astar_weighted
else:
    from weighted_astar import astar_weighted

Coord = Tuple[int, int]
NAMES = ("A", "B", "T1", "T2", "T3", "T4")
POSITIONS: Dict[str, Coord] = {
    "A": (0, 0), "B": (6, 6), "T1": (0, 6),
    "T2": (2, 2), "T3": (4, 4), "T4": (6, 0),
}
TREASURE_VALUES = {"T1": 9, "T2": 6, "T3": 7, "T4": 7}

# 逐格对照实验的7×7地图。
MAP: List[List[object]] = [
    ["A", 1, 1, "X", 2, 1, "T1"],
    [1, "X", 6, "X", 1, "X", 1],
    [1, 1, "T2", 1, 1, 1, 1],
    ["X", 1, "X", 8, "X", 1, "X"],
    [1, 1, 1, 1, "T3", 1, 1],
    [1, "X", 9, "X", 1, "X", 1],
    ["T4", 1, 1, "X", 1, 1, "B"],
]
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def validate_map() -> None:
    if len(MAP) != 7 or any(len(row) != 7 for row in MAP):
        raise ValueError("地图必须是7×7规则矩形")
    for name, (r, c) in POSITIONS.items():
        if MAP[r][c] != name:
            raise ValueError(
                f"地图标记不符: {name}应位于{(r, c)}，实际为{MAP[r][c]!r}"
            )
    for row in MAP:
        for cell in row:
            valid = (
                cell == "X"
                or cell in NAMES
                or (type(cell) is int and cell >= 1)
            )
            if not valid:
                raise ValueError(f"题目四地图出现非法格子: {cell!r}")


def real_entry_cost(cell: object) -> int:
    """题目四中，特殊点A/B/T1~T4均按代价1处理。"""
    if cell == "X":
        raise ValueError("不能进入障碍物")
    return 1 if isinstance(cell, str) else int(cell)


def converted_grid(start: str, goal: str) -> List[List[object]]:
    """为复用题目二的A*，只在临时网格中将目标格置为免费E。"""
    grid = [[1 if cell in NAMES else cell for cell in row] for row in MAP]
    sr, sc = POSITIONS[start]
    gr, gc = POSITIONS[goal]
    grid[sr][sc] = "S"
    grid[gr][gc] = "E"
    return grid


def actual_path_cost(path: List[Coord]) -> int:
    """独立按题目四原始格子权值累计：不计起点，计入终点。"""
    return sum(real_entry_cost(MAP[r][c]) for r, c in path[1:])


def independent_dijkstra(start: Coord, goal: Coord) -> int:
    """独立最短路径基准（不依赖 weighted_astar.py），用于正确性检查。"""
    heap = [(0, start)]
    best = {start: 0}
    while heap:
        cost, current = heappop(heap)
        if cost != best[current]:
            continue
        if current == goal:
            return cost
        r, c = current
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < 7 and 0 <= nc < 7) or MAP[nr][nc] == "X":
                continue
            point = (nr, nc)
            new_cost = cost + real_entry_cost(MAP[nr][nc])
            if new_cost < best.get(point, inf):
                best[point] = new_cost
                heappush(heap, (new_cost, point))
    return -1


def build_distance_matrix() -> Tuple[Dict[str, Dict[str, int]], List[dict]]:
    """逐个有序点对用A*求距离，返回距离字典与30条搜索统计。"""
    validate_map()
    matrix: Dict[str, Dict[str, int]] = {name: {} for name in NAMES}
    stats: List[dict] = []

    for src in NAMES:
        for dst in NAMES:
            if src == dst:
                matrix[src][dst] = 0
                continue
            start, goal = POSITIONS[src], POSITIONS[dst]
            # 本题目标格代价为1；临时E代价为0，故最优路径代价统一+1。
            result = astar_weighted(start, goal, converted_grid(src, dst), "adjusted_manhattan")
            if result.path_cost is None:
                raise ValueError(f"固定地图存在不可达的关键点：{src}->{dst}")
            distance = result.path_cost + 1
            assert result.path[0] == start and result.path[-1] == goal
            assert all(abs(a[0]-b[0]) + abs(a[1]-b[1]) == 1
                       for a, b in zip(result.path, result.path[1:]))
            assert distance == actual_path_cost(result.path), f"路径累计代价有误: {src}->{dst}"
            expected = independent_dijkstra(start, goal)
            assert distance == expected, f"与独立Dijkstra不一致: {src}->{dst}, {distance}!={expected}"
            matrix[src][dst] = distance
            stats.append({
                "start": src, "goal": dst, "min_cost": distance,
                "steps": len(result.path) - 1,
                "expanded_nodes": result.expanded_nodes,
                "generated_nodes": result.generated_nodes,
                "elapsed_ms": round(result.elapsed_seconds * 1000, 6),
            })

    # 这6个点的进入权重都是1，反向路径在当前规则下代价一致。
    for a in NAMES:
        for b in NAMES:
            assert matrix[a][b] == matrix[b][a], f"距离矩阵不对称: {a}, {b}"
    assert matrix["A"]["T2"] == 4
    assert matrix["B"]["T3"] == 4
    assert matrix["A"]["T3"] == 8
    return matrix, stats


def save_results(matrix: Dict[str, Dict[str, int]], stats: List[dict]) -> None:
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    matrix_path = out_dir / "q4_distance_matrix.csv"
    with matrix_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["from\\to", *NAMES])
        for src in NAMES:
            writer.writerow([src, *(matrix[src][dst] for dst in NAMES)])
    stats_path = out_dir / "q4_pairwise_astar_stats.csv"
    with stats_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats[0].keys()))
        writer.writeheader()
        writer.writerows(stats)
    print(f"距离矩阵CSV：{matrix_path}")
    print(f"A*搜索统计CSV：{stats_path}")


def main() -> None:
    matrix, stats = build_distance_matrix()
    print("=== 实验二 / 题目四 第1阶段：6×6 A*最小移动代价矩阵 ===")
    print("     " + " ".join(f"{name:>4}" for name in NAMES))
    for src in NAMES:
        print(f"{src:>4} " + " ".join(f"{matrix[src][dst]:>4}" for dst in NAMES))
    print("\n[PASS] 30组有序不同点对 A* 路径合法、代价一致，均与独立Dijkstra结果匹配")
    print("[PASS] 对角线为0，矩阵对称，关键距离A->T2=4、B->T3=4、A->T3=8")
    print("说明：与题目二不同，题目四进入宝藏格的代价为1，不是0。")
    save_results(matrix, stats)
    print("=== 第四题第1阶段检查全部通过 ===")


if __name__ == "__main__":
    main()
