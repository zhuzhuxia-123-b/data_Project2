from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Dict, List, Sequence, Tuple

if __package__:
    from .treasure_distance import (
        NAMES,
        POSITIONS,
        build_distance_matrix,
        converted_grid,
    )
    from .treasure_game import initial_state, minimax_search
    from .weighted_astar import AStarResult, astar_weighted
else:
    from treasure_distance import (
        NAMES,
        POSITIONS,
        build_distance_matrix,
        converted_grid,
    )
    from treasure_game import initial_state, minimax_search
    from weighted_astar import AStarResult, astar_weighted


REPEATS = 30
WARMUPS = 3
WORKLOADS: Tuple[Tuple[str, Tuple[int, ...]], ...] = (
    ("H2", (2,)),
    ("H4", (4,)),
    ("H2_H4_REUSE", (2, 4)),
)

DistanceMatrix = Dict[str, Dict[str, int]]


@dataclass
class AStarUsage:
    calls: int = 0
    expanded_nodes: int = 0
    generated_nodes: int = 0
    elapsed_ms: float = 0.0

    def record(self, result: AStarResult) -> None:
        self.calls += 1
        self.expanded_nodes += result.expanded_nodes
        self.generated_nodes += result.generated_nodes
        self.elapsed_ms += result.elapsed_seconds * 1000


def compute_pair_distance(source: str, target: str,
                          usage: AStarUsage) -> int:
    """使用与正式距离矩阵相同的 A* 规则计算一个有序点对。"""
    if source not in POSITIONS or target not in POSITIONS:
        raise KeyError(f"未知关键位置: {source}->{target}")
    if source == target:
        return 0

    result = astar_weighted(
        POSITIONS[source],
        POSITIONS[target],
        converted_grid(source, target),
        heuristic_mode="adjusted_manhattan",
    )
    if result.path_cost is None:
        raise ValueError(f"关键位置不可达: {source}->{target}")
    usage.record(result)

    # 临时网格中的 E 免费；题目四进入目标关键点的真实代价为1。
    return result.path_cost + 1


def precompute_distances() -> Tuple[DistanceMatrix, AStarUsage]:
    """一次性计算全部30个非对角有序点对。"""
    usage = AStarUsage()
    matrix: DistanceMatrix = {source: {} for source in NAMES}
    for source in NAMES:
        for target in NAMES:
            matrix[source][target] = compute_pair_distance(
                source, target, usage,
            )
    if usage.calls != len(NAMES) * (len(NAMES) - 1):
        raise AssertionError(f"预计算A*调用数异常: {usage.calls}")
    return matrix, usage


class _OnDemandRow:
    def __init__(self, owner: "OnDemandDistances", source: str):
        self.owner = owner
        self.source = source

    def __getitem__(self, target: str) -> int:
        value = compute_pair_distance(self.source, target, self.owner.usage)
        self.owner.observed.append((self.source, target, value))
        return value


class OnDemandDistances:
    """模拟距离矩阵接口，但每次查询都重新运行A*且不缓存。"""

    def __init__(self):
        self.usage = AStarUsage()
        self.observed: List[Tuple[str, str, int]] = []

    def __getitem__(self, source: str) -> _OnDemandRow:
        if source not in POSITIONS:
            raise KeyError(source)
        return _OnDemandRow(self, source)


@dataclass
class TrialResult:
    decisions: str
    utilities: str
    visited_states: int
    evaluated_leaves: int
    astar_calls: int
    astar_expanded_nodes: int
    astar_generated_nodes: int
    astar_elapsed_ms: float
    search_phase_ms: float
    total_elapsed_ms: float


def _encode_outcomes(results: Sequence[object]) -> Tuple[str, str]:
    decisions = ";".join(str(result.best_target) for result in results)
    utilities = ";".join(str(result.best_value) for result in results)
    return decisions, utilities


def run_trial(strategy: str, depths: Sequence[int],
              reference: DistanceMatrix) -> TrialResult:
    """执行一次完整工作负载；正确性检查放在计时区间之外。"""
    state = initial_state()
    total_started = perf_counter()

    if strategy == "precomputed":
        distances, usage = precompute_distances()
        search_started = perf_counter()
        results = [
            minimax_search(state, distances, depth, "left_to_right")
            for depth in depths
        ]
        search_phase_ms = (perf_counter() - search_started) * 1000
        observed = None
    elif strategy == "on_demand":
        distances = OnDemandDistances()
        usage = distances.usage
        search_started = perf_counter()
        results = [
            minimax_search(state, distances, depth, "left_to_right")
            for depth in depths
        ]
        search_phase_ms = (perf_counter() - search_started) * 1000
        observed = distances.observed
    else:
        raise ValueError("strategy只能是precomputed或on_demand")

    total_elapsed_ms = (perf_counter() - total_started) * 1000

    # 正确性核验不计入策略总耗时。
    if strategy == "precomputed":
        if distances != reference:
            raise AssertionError("预计算距离矩阵与独立验证矩阵不一致")
    else:
        for source, target, value in observed or ():
            if value != reference[source][target]:
                raise AssertionError(
                    f"按需距离错误: {source}->{target}, "
                    f"得到{value}, 期望{reference[source][target]}"
                )

    expected = {2: ("T2", -1), 4: ("T3", 3)}
    for depth, result in zip(depths, results):
        if (result.best_target, result.best_value) != expected[depth]:
            raise AssertionError(
                f"H={depth}结果错误: "
                f"{result.best_target}/{result.best_value}"
            )

    # 完整Minimax每条被访问的父子边恰好触发一次距离查询。
    expected_queries = sum(result.visited_states - 1 for result in results)
    if strategy == "on_demand" and usage.calls != expected_queries:
        raise AssertionError(
            f"按需A*调用数异常: {usage.calls} != {expected_queries}"
        )

    decisions, utilities = _encode_outcomes(results)
    return TrialResult(
        decisions=decisions,
        utilities=utilities,
        visited_states=sum(result.visited_states for result in results),
        evaluated_leaves=sum(result.evaluated_leaves for result in results),
        astar_calls=usage.calls,
        astar_expanded_nodes=usage.expanded_nodes,
        astar_generated_nodes=usage.generated_nodes,
        astar_elapsed_ms=usage.elapsed_ms,
        search_phase_ms=search_phase_ms,
        total_elapsed_ms=total_elapsed_ms,
    )


def benchmark(workload: str, depths: Tuple[int, ...], strategy: str,
              reference: DistanceMatrix) -> dict:
    for _ in range(WARMUPS):
        run_trial(strategy, depths, reference)
    trials = [
        run_trial(strategy, depths, reference)
        for _ in range(REPEATS)
    ]
    first = trials[0]
    deterministic = (
        first.decisions,
        first.utilities,
        first.visited_states,
        first.evaluated_leaves,
        first.astar_calls,
        first.astar_expanded_nodes,
        first.astar_generated_nodes,
    )
    for trial in trials[1:]:
        current = (
            trial.decisions,
            trial.utilities,
            trial.visited_states,
            trial.evaluated_leaves,
            trial.astar_calls,
            trial.astar_expanded_nodes,
            trial.astar_generated_nodes,
        )
        if current != deterministic:
            raise AssertionError(
                f"{workload}/{strategy}的确定性结果不稳定"
            )

    return {
        "workload": workload,
        "depths": ";".join(map(str, depths)),
        "strategy": strategy,
        "decisions": first.decisions,
        "utilities": first.utilities,
        "visited_states": first.visited_states,
        "evaluated_leaves": first.evaluated_leaves,
        "astar_calls": first.astar_calls,
        "astar_expanded_nodes": first.astar_expanded_nodes,
        "astar_generated_nodes": first.astar_generated_nodes,
        "median_astar_elapsed_ms": round(
            median(trial.astar_elapsed_ms for trial in trials), 6,
        ),
        "median_search_phase_ms": round(
            median(trial.search_phase_ms for trial in trials), 6,
        ),
        "median_total_elapsed_ms": round(
            median(trial.total_elapsed_ms for trial in trials), 6,
        ),
        "warmups": WARMUPS,
        "repeats": REPEATS,
    }


def main() -> None:
    print("=== 创新对照：距离矩阵预计算 vs 按需重复A* ===")
    print(
        f"每种配置预热{WARMUPS}次、正式运行{REPEATS}次，"
        "耗时取中位数。"
    )
    print("正确性基准只构造一次，不计入两种策略的计时。")

    reference, _ = build_distance_matrix()
    rows: List[dict] = []
    for workload, depths in WORKLOADS:
        print(f"\n[{workload}] depths={depths}")
        group = [
            benchmark(workload, depths, strategy, reference)
            for strategy in ("precomputed", "on_demand")
        ]
        if (
            group[0]["decisions"], group[0]["utilities"]
        ) != (
            group[1]["decisions"], group[1]["utilities"]
        ):
            raise AssertionError(f"{workload}两种策略的博弈结果不一致")

        rows.extend(group)
        for row in group:
            print(
                f"  {row['strategy']:<11} "
                f"结果={row['decisions']}/{row['utilities']} "
                f"A*调用={row['astar_calls']:>2} "
                f"展开={row['astar_expanded_nodes']:>4} "
                f"总耗时中位数={row['median_total_elapsed_ms']:.4f}ms"
            )

        call_delta = group[1]["astar_calls"] - group[0]["astar_calls"]
        if call_delta > 0:
            print(f"  预计算少运行 {call_delta} 次A*。")
        elif call_delta < 0:
            print(f"  单次浅层搜索按需方式少运行 {-call_delta} 次A*。")
        else:
            print("  两种策略的A*调用次数相同。")

    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "q4_precomputation_comparison.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    if len(rows) != len(WORKLOADS) * 2:
        raise AssertionError("预计算对照CSV行数不正确")
    print(f"\n[PASS] 两种策略的H=2/H=4决策与收益完全一致")
    print(f"已保存 {len(rows)} 行对照数据：{output}")
    print("解释时优先使用A*调用数和展开节点；短时间计时可能波动。")


if __name__ == "__main__":
    main()

