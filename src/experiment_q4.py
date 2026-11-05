"""当代人工智能实验二 / 题目四 第3阶段：博弈搜索对照实验。

依赖（放在同一文件夹）：
    treasure_game.py, treasure_distance.py, weighted_astar.py, grid_search.py
运行：
    python experiment_q4.py

实验目标：
1. 对比 H=1、2、3、4 的决策和收益差（H=2、H=4 是PDF必测深度）。
2. 比较 Minimax / Alpha-Beta、从左到右 / 从右到左的搜索开销。
3. 不修改已有算法，独立枚举完整的65个可达博弈状态并交叉核验。
4. 导出两张可复现的CSV：搜索性能表与根节点候选行动收益表。

计时说明：每个配置120次取中位数，计时不含A*距离矩阵预计算。
不同访问顺序遇到相同最优值时可以选择不同的并列首步；
所以以最佳收益差一致、且首步属于最优行动集合为正确性准则。
"""

import csv
from pathlib import Path
from statistics import median
from typing import Dict, List, Set, Tuple

if __package__:
    from .treasure_distance import TREASURE_VALUES, build_distance_matrix
    from .treasure_game import (
        GameState, alpha_beta_search, initial_state, minimax_search, take_treasure,
    )
else:
    from treasure_distance import TREASURE_VALUES, build_distance_matrix
    from treasure_game import (
        GameState, alpha_beta_search, initial_state, minimax_search, take_treasure,
    )

REPEATS = 120
ORDERS = ("left_to_right", "right_to_left")
ALGORITHMS = (("Minimax", minimax_search), ("Alpha-Beta", alpha_beta_search))
DEPTHS = (1, 2, 3, 4)


def independent_reference(state: GameState, distances: Dict[str, Dict[str, int]],
                          depth: int) -> int:
    """独立实现极大极小递推，仅用来校验实际搜索返回的博弈值。"""
    if depth == 0 or not state.remaining:
        return state.score_a - state.score_b
    candidates: List[int] = []
    for target in state.remaining:
        child, _ = take_treasure(state, target, distances)
        candidates.append(independent_reference(child, distances, depth - 1))
    return max(candidates) if state.current_player == "A" else min(candidates)


def independent_optimal_actions(state: GameState,
                                distances: Dict[str, Dict[str, int]],
                                depth: int) -> Set[str]:
    """对并列最优动作保持宽容：顺序不同允许返回不同但等价的动作。"""
    if depth == 0 or not state.remaining:
        return set()
    action_values = {}
    for action in state.remaining:
        child, _ = take_treasure(state, action, distances)
        action_values[action] = independent_reference(child, distances, depth - 1)
    best = max(action_values.values()) if state.current_player == "A" else min(action_values.values())
    return {action for action, value in action_values.items() if value == best}


def reachable_states(state: GameState, distances: Dict[str, Dict[str, int]]) -> Set[GameState]:
    """从给定初态遍历四轮游戏的所有65个可达状态（含根和叶）。"""
    seen: Set[GameState] = set()

    def explore(s: GameState) -> None:
        if s in seen:
            return
        seen.add(s)
        for target in s.remaining:
            child, _ = take_treasure(s, target, distances)
            explore(child)

    explore(state)
    return seen


def verify_all_reachable_states(distances: Dict[str, Dict[str, int]]) -> Tuple[int, int]:
    """对每个可达状态，检查全部合法剩余深度及两种访问顺序。"""
    all_states = reachable_states(initial_state(), distances)
    assert len(all_states) == 65, f"期望65个可达状态，实际{len(all_states)}"
    checked = 0
    for state in all_states:
        for depth in range(len(state.remaining) + 1):
            expected = independent_reference(state, distances, depth)
            optimal = independent_optimal_actions(state, distances, depth)
            for order in ORDERS:
                mm = minimax_search(state, distances, depth, order)
                ab = alpha_beta_search(state, distances, depth, order)
                assert mm.best_value == ab.best_value == expected, (
                    f"搜索值不一致: state={state}, depth={depth}, order={order}; "
                    f"reference={expected}, minimax={mm.best_value}, alpha-beta={ab.best_value}"
                )
                assert mm.best_target in optimal if optimal else mm.best_target is None
                assert ab.best_target in optimal if optimal else ab.best_target is None
                assert ab.visited_states <= mm.visited_states
                assert ab.evaluated_leaves <= mm.evaluated_leaves
                checked += 1
    return len(all_states), checked


def bench_once(state: GameState, distances: Dict[str, Dict[str, int]],
               depth: int, name: str, search_func, order: str) -> dict:
    """执行REPEATS次，汇总确定性节点指标和运行时间中位数。"""
    first = search_func(state, distances, depth, order)
    durations = [first.elapsed_ms]
    for _ in range(REPEATS - 1):
        result = search_func(state, distances, depth, order)
        assert (result.best_target, result.best_value, result.visited_states,
                result.evaluated_leaves, result.cutoff_events, result.skipped_actions,
                result.max_depth_reached) == (
                    first.best_target, first.best_value, first.visited_states,
                    first.evaluated_leaves, first.cutoff_events, first.skipped_actions,
                    first.max_depth_reached
                ), "相同配置的搜索输出不稳定"
        durations.append(result.elapsed_ms)
    return {
        "depth": depth,
        "algorithm": name,
        "order": order,
        "best_target": first.best_target or "",
        "best_value": first.best_value,
        "visited_states": first.visited_states,
        "max_depth_reached": first.max_depth_reached,
        "evaluated_leaves": first.evaluated_leaves,
        "cutoff_events": first.cutoff_events,
        "skipped_actions": first.skipped_actions,
        "reduction_vs_minimax_pct": 0.0,  # 在完整分组后填充
        "median_elapsed_ms": round(median(durations), 6),
        "repeats": REPEATS,
    }


def compute_root_action_values(distances: Dict[str, Dict[str, int]]) -> List[dict]:
    """核对所有首步的H=2/H=4价值，解释为什么首步从T2变成T3。"""
    state = initial_state()
    rows = []
    for depth in (2, 4):
        ref_best = independent_reference(state, distances, depth)
        for target in state.remaining:
            child, gain = take_treasure(state, target, distances)
            value = independent_reference(child, distances, depth - 1)
            rows.append({
                "depth": depth,
                "first_target": target,
                "first_gain": gain,
                "final_minimax_utility": value,
                "is_optimal": int(value == ref_best),
            })
    return rows


def save_csv(path: Path, rows: List[dict]) -> None:
    assert rows
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    print("=== 实验二 / 题目四 第3阶段：博弈搜索性能对照 ===")
    print(f"每种配置重复{REPEATS}次，耗时取中位数（ms）；距离矩阵只预计算一次。")
    distances, _ = build_distance_matrix()
    state = initial_state()
    rows: List[dict] = []

    for depth in DEPTHS:
        expected = independent_reference(state, distances, depth)
        print(f"\n[H={depth}] 独立枚举参考最优收益差={expected}")
        for order in ORDERS:
            mini = bench_once(state, distances, depth, "Minimax", minimax_search, order)
            alpha = bench_once(state, distances, depth, "Alpha-Beta", alpha_beta_search, order)
            assert mini["best_value"] == alpha["best_value"] == expected
            optimal = independent_optimal_actions(state, distances, depth)
            assert mini["best_target"] in optimal and alpha["best_target"] in optimal
            assert alpha["visited_states"] <= mini["visited_states"]
            assert alpha["evaluated_leaves"] <= mini["evaluated_leaves"]
            assert mini["max_depth_reached"] == alpha["max_depth_reached"] == depth
            saving = 100.0 * (mini["visited_states"] - alpha["visited_states"]) / mini["visited_states"]
            alpha["reduction_vs_minimax_pct"] = round(saving, 2)
            for result in (mini, alpha):
                print(f"  {result['algorithm']:<10} {order:<13} 首步={result['best_target']} "
                      f"U={result['best_value']:+d} 访问={result['visited_states']:>2} "
                      f"叶子={result['evaluated_leaves']:>2} "
                      f"剪枝={result['cutoff_events']:>2} "
                      f"耗时中位数={result['median_elapsed_ms']:.4f}ms")
                rows.append(result)
            print(f"  {order}：Alpha-Beta减少访问状态 {saving:.1f}%")
        print(f"  [PASS] H={depth}，两种算法×两种顺序均符合独立枚举结果")

    # 强制核对PDF中的两组重点预期。
    for depth, target, value in ((2, "T2", -1), (4, "T3", 3)):
        group = [r for r in rows if r["depth"] == depth]
        assert len(group) == 4
        assert all(r["best_target"] == target and r["best_value"] == value for r in group)
    print("\n[PASS] PDF必测结论 H=2 -> T2/-1, H=4 -> T3/+3")

    print("\n=== H=2与H=4：所有首步的博弈收益对照 ===")
    first_actions = compute_root_action_values(distances)
    for depth in (2, 4):
        print(f"  [H={depth}]")
        for record in first_actions:
            if record["depth"] == depth:
                print(f"    {record['first_target']}: 首回合Gain={record['first_gain']:+d}, "
                      f"最优应对后的U={record['final_minimax_utility']:+d}" +
                      ("  <- 最优首步" if record["is_optimal"] else ""))

    print("\n=== 全可达状态交叉验证 ===")
    count_states, checked = verify_all_reachable_states(distances)
    print(f"[PASS] {count_states}个可达状态；{checked}组状态/深度/顺序检查：")
    print("       Minimax与Alpha-Beta价值一致，且最优行动与独立枚举相符")

    result_dir = Path(__file__).resolve().parent / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    path_performance = result_dir / "q4_game_search_comparison.csv"
    path_first_action = result_dir / "q4_first_action_values.csv"
    save_csv(path_performance, rows)
    save_csv(path_first_action, first_actions)
    assert len(rows) == 16 and len(first_actions) == 8
    print(f"\n[PASS] 已导出16行性能对照数据：{path_performance}")
    print(f"[PASS] 已导出8行候选首步数据：{path_first_action}")
    print("注意：节点访问数是确定性指标；短时耗时受机器负载影响。")
    print("=== 第四题第3阶段：全部检查通过 ===")


if __name__ == "__main__":
    main()
