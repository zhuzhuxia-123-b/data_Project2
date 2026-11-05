"""当代人工智能实验二 / 适度创新：基于即时净收益的 Alpha-Beta 动态动作排序。

把此文件与 treasure_game.py、treasure_distance.py、experiment_q4.py 等放同一 src 目录。
运行：python experiment_innovation.py

方法（无信息泄露）：
  每个内部节点，只根据当前玩家所在位置和已预计算的 A* 距离，
  计算各合法宝藏的即时净收益 gain = value - distance。
  将收益高的宝藏优先搜索（对玩家A/B都排序为收益降序）；平局保持T1..T4的原顺序。
  A 使 Score_A-Score_B 最大；B 使该差最小，优先增加自己的收益。
  排序只是搜索先后次序，不改变状态转移、目标函数、截断深度与正确的博弈值。

注意：这是低成本启发式，并非必然优于固定访问顺序；排序成本计入运行耗时。

输出：results/q4_gain_ordering_comparison.csv
"""

import csv
from math import inf
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Dict, List, Optional, Tuple

if __package__:
    from .treasure_distance import TREASURE_VALUES, build_distance_matrix
    from .treasure_game import (
        GameSearchResult,
        GameState,
        alpha_beta_search,
        initial_state,
        minimax_search,
        take_treasure,
    )
    from .experiment_q4 import (
        independent_optimal_actions,
        independent_reference,
        reachable_states,
    )
else:
    from treasure_distance import TREASURE_VALUES, build_distance_matrix
    from treasure_game import (
        GameSearchResult,
        GameState,
        alpha_beta_search,
        initial_state,
        minimax_search,
        take_treasure,
    )
    from experiment_q4 import (
        independent_optimal_actions,
        independent_reference,
        reachable_states,
    )

REPEATS = 120
DistanceMatrix = Dict[str, Dict[str, int]]


def gain_sorted_actions(state: GameState, distances: DistanceMatrix) -> Tuple[str, ...]:
    """只使用当前状态已知信息，不读取未来子树/完整Minimax结果。"""
    source = state.pos_a if state.current_player == "A" else state.pos_b
    # Python稳定排序：即时净收益相同时，保留state.remaining中的原始次序。
    for target in state.remaining:
        if distances[source][target] < 0:
            raise ValueError(f"不可达的宝藏: {source}->{target}")
    return tuple(sorted(
        state.remaining,
        key=lambda t: -(TREASURE_VALUES[t] - distances[source][t]),
    ))


def alpha_beta_gain_ordered(state: GameState, distances: DistanceMatrix,
                            depth: int) -> GameSearchResult:
    """Alpha-Beta + 每层即时净收益降序动作排序。耗时包含排序。"""
    if not (isinstance(depth, int) and 0 <= depth <= 4):
        raise ValueError("本题搜索深度必须介于0和4之间")
    visited = leaves = cutoffs = skipped = max_depth = 0

    def solve(s: GameState, h: int, alpha: float, beta: float,
              current_depth: int) -> Tuple[int, Optional[str]]:
        nonlocal visited, leaves, cutoffs, skipped, max_depth
        visited += 1
        max_depth = max(max_depth, current_depth)
        if h == 0 or not s.remaining:
            leaves += 1
            return s.utility, None

        maximizing = s.current_player == "A"
        best = -inf if maximizing else inf
        best_target = None
        actions = gain_sorted_actions(s, distances)
        for index, target in enumerate(actions):
            nxt, _ = take_treasure(s, target, distances)
            value, _ = solve(nxt, h - 1, alpha, beta, current_depth + 1)
            if best_target is None or (maximizing and value > best) or (
                not maximizing and value < best
            ):
                best, best_target = value, target
            if maximizing:
                alpha = max(alpha, best)
            else:
                beta = min(beta, best)
            if alpha >= beta:
                remaining_count = len(actions) - index - 1
                if remaining_count:
                    cutoffs += 1
                    skipped += remaining_count
                break
        return int(best), best_target

    start = perf_counter()
    best_value, best_target = solve(state, depth, -inf, inf, 0)
    elapsed = (perf_counter() - start) * 1000
    return GameSearchResult(best_target, best_value, visited, leaves, cutoffs, skipped,
                            elapsed, max_depth_reached=max_depth)


def verify_all_states(distances: DistanceMatrix) -> Tuple[int, int]:
    """65个可达状态，枚举所有合法深度，与独立枚举结果比较。"""
    states = reachable_states(initial_state(), distances)
    assert len(states) == 65
    checks = 0
    for state in states:
        for depth in range(len(state.remaining) + 1):
            expected = independent_reference(state, distances, depth)
            optimal = independent_optimal_actions(state, distances, depth)
            ordered = alpha_beta_gain_ordered(state, distances, depth)
            baseline = minimax_search(state, distances, depth)
            assert ordered.best_value == baseline.best_value == expected, (
                f"最优值不一致: {state}, H={depth}, reference={expected}, result={ordered}"
            )
            if optimal:
                assert ordered.best_target in optimal, (
                    f"首步不是最优动作: {state}, H={depth}, target={ordered.best_target}, "
                    f"optimal={optimal}"
                )
            else:
                assert ordered.best_target is None
            assert ordered.max_depth_reached == baseline.max_depth_reached == depth
            assert ordered.visited_states <= baseline.visited_states
            assert ordered.evaluated_leaves <= baseline.evaluated_leaves
            checks += 1
    return len(states), checks


def measure(name: str, func, state: GameState,
            distances: DistanceMatrix, depth: int) -> dict:
    first = func(state, distances, depth)
    times = [first.elapsed_ms]
    expected_signature = (
        first.best_target, first.best_value, first.visited_states,
        first.evaluated_leaves, first.cutoff_events, first.skipped_actions,
        first.max_depth_reached,
    )
    for _ in range(REPEATS - 1):
        result = func(state, distances, depth)
        signature = (
            result.best_target, result.best_value, result.visited_states,
            result.evaluated_leaves, result.cutoff_events, result.skipped_actions,
            result.max_depth_reached,
        )
        assert signature == expected_signature, f"配置{name}/H={depth}结果不稳定"
        times.append(result.elapsed_ms)
    return {
        "depth": depth,
        "algorithm": name,
        "best_target": first.best_target or "",
        "best_value": first.best_value,
        "visited_states": first.visited_states,
        "max_depth_reached": first.max_depth_reached,
        "evaluated_leaves": first.evaluated_leaves,
        "cutoff_events": first.cutoff_events,
        "skipped_actions": first.skipped_actions,
        "reduction_vs_left_to_right_pct": 0.0,
        "median_elapsed_ms": round(median(times), 6),
        "repeats": REPEATS,
    }


def main() -> None:
    print("=== 实验二 / 创新拓展：即时收益优先的 Alpha-Beta 动作排序 ===")
    print("设计：每层按当前行动者的即时净收益降序；不窥视未来Minimax结果。")
    print(f"每种配置运行{REPEATS}次，耗时取中位数（ms）；动态排序的时间开销计入计时。")
    distances, _ = build_distance_matrix()
    state = initial_state()

    expected_start_order = gain_sorted_actions(state, distances)
    assert expected_start_order == ("T2", "T1", "T3", "T4"), expected_start_order
    print("初始A的动作顺序（平局保持原序）：", " -> ".join(expected_start_order))

    configurations = (
        ("Minimax_fixed", lambda s, d, h: minimax_search(s, d, h, "left_to_right")),
        ("AlphaBeta_left_to_right", lambda s, d, h: alpha_beta_search(s, d, h, "left_to_right")),
        ("AlphaBeta_right_to_left", lambda s, d, h: alpha_beta_search(s, d, h, "right_to_left")),
        ("AlphaBeta_gain_desc", alpha_beta_gain_ordered),
    )
    rows: List[dict] = []
    for depth in (1, 2, 3, 4):
        expected = independent_reference(state, distances, depth)
        optimal = independent_optimal_actions(state, distances, depth)
        print(f"\n[H={depth}] 独立枚举最优收益差={expected}")
        results = [measure(name, func, state, distances, depth)
                   for name, func in configurations]
        fixed = results[1]["visited_states"]
        for row in results:
            assert row["best_value"] == expected
            assert row["best_target"] in optimal
            row["reduction_vs_left_to_right_pct"] = round(
                100.0 * (fixed - row["visited_states"]) / fixed, 2
            )
            rows.append(row)
            print(f"  {row['algorithm']:<25} 首步={row['best_target']} "
                  f"U={row['best_value']:+d} 访问={row['visited_states']:>2} "
                  f"叶子={row['evaluated_leaves']:>2} "
                  f"耗时中位数={row['median_elapsed_ms']:.4f}ms")
        gain = results[-1]
        left = results[1]
        diff = left["visited_states"] - gain["visited_states"]
        pct = 100.0 * diff / left["visited_states"]
        print(f"  动态排序相对固定左到右：少访问{diff}个状态（{pct:.1f}%）")

    assert (rows[7]["best_target"], rows[7]["best_value"]) == ("T2", -1)
    assert (rows[15]["best_target"], rows[15]["best_value"]) == ("T3", 3)
    print("\n=== 全可达状态回归 ===")
    states_count, checks = verify_all_states(distances)
    print(f"[PASS] {states_count}个可达状态、{checks}组状态/深度检查：")
    print("       动态排序与独立Minimax最优值一致，且所选首步属于最优动作集合")
    print("[PASS] PDF H=2：T2/-1；H=4：T3/+3")

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "q4_gain_ordering_comparison.csv"
    with output_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert len(rows) == 16
    print(f"[PASS] 已保存16行对照数据：{output_path}")
    print("注意：小规模搜索中计时存在波动，是否更快应以实测结果为准；节点数更稳定。")
    print("=== 创新拓展：正确性及性能对照全部通过 ===")


if __name__ == "__main__":
    main()
