"""当代人工智能实验二·题目四（第2阶段）：宝藏分配的 Minimax / Alpha-Beta。

依赖：同目录的 treasure_distance.py、weighted_astar.py、grid_search.py。
运行：python treasure_game.py

题意：A(MAX)→B(MIN)→A→B；取宝藏时 Gain=Vi-最小移动代价；
经过别的宝藏不自动收集；取完的宝藏格仍可通过；不禁止负收益。
本阶段验证 H=2 与 H=4 的首步及收益差；性能对照留待下一阶段。
"""

from dataclasses import dataclass
from math import inf
from time import perf_counter
from typing import Dict, Optional, Tuple

if __package__:
    from .treasure_distance import NAMES, TREASURE_VALUES, build_distance_matrix
else:
    from treasure_distance import NAMES, TREASURE_VALUES, build_distance_matrix

DistanceMatrix = Dict[str, Dict[str, int]]
TREASURES = tuple(name for name in NAMES if name.startswith("T"))


@dataclass(frozen=True)
class GameState:
    """不可变状态，避免递归回溯时相互污染。"""
    pos_a: str
    pos_b: str
    remaining: Tuple[str, ...]
    score_a: int
    score_b: int
    current_player: str  # "A" or "B"

    @property
    def utility(self) -> int:
        return self.score_a - self.score_b


def initial_state() -> GameState:
    return GameState("A", "B", TREASURES, 0, 0, "A")


def take_treasure(state: GameState, target: str,
                  distances: DistanceMatrix) -> Tuple[GameState, int]:
    """选取一个未获得的宝藏，更新玩家位置、收益、剩余宝藏和回合。"""
    if target not in state.remaining:
        raise ValueError(f"宝藏不存在或已取得: {target}")
    if state.current_player not in ("A", "B"):
        raise ValueError("当前玩家只能是 A 或 B")

    others = tuple(t for t in state.remaining if t != target)
    source = state.pos_a if state.current_player == "A" else state.pos_b
    # 严禁将 -1/inf/缺失视为负移动成本；只接受有限、非负的数字距离。
    try:
        distance = distances[source][target]
    except KeyError as exc:
        raise ValueError(f"缺少距离或不可到达的宝藏：{source} -> {target}") from exc
    if (not isinstance(distance, (int, float)) or isinstance(distance, bool)
            or not (0 <= distance < inf)):
        raise ValueError(f"不可到达或非法的移动代价：{source} -> {target} = {distance!r}")
    gain = TREASURE_VALUES[target] - distance  # 可以为负数，不得截断为0

    if state.current_player == "A":
        return GameState(target, state.pos_b, others,
                         state.score_a + gain, state.score_b, "B"), gain
    return GameState(state.pos_a, target, others,
                     state.score_a, state.score_b + gain, "A"), gain


@dataclass
class GameSearchResult:
    best_target: Optional[str]
    best_value: int
    visited_states: int
    evaluated_leaves: int
    cutoff_events: int
    skipped_actions: int
    elapsed_ms: float
    max_depth_reached: int = 0  # 根状态深度0，实际进入递归的最大深度


def _actions(state: GameState, order: str) -> Tuple[str, ...]:
    if order == "left_to_right":
        return state.remaining
    if order == "right_to_left":
        return tuple(reversed(state.remaining))
    raise ValueError("order必须是left_to_right或right_to_left")


def minimax_search(state: GameState, distances: DistanceMatrix,
                   depth: int, order: str = "left_to_right") -> GameSearchResult:
    """完全枚举给定深度内的所有合法回合，A最大化U，B最小化U。"""
    if not (isinstance(depth, int) and 0 <= depth <= 4):
        raise ValueError("本题搜索深度必须介于0和4之间")
    _actions(state, order)  # 及早检查order
    visited = leaves = max_depth = 0

    def solve(s: GameState, remaining_depth: int,
              current_depth: int) -> Tuple[int, Optional[str]]:
        nonlocal visited, leaves, max_depth
        visited += 1
        max_depth = max(max_depth, current_depth)
        if remaining_depth == 0 or not s.remaining:
            leaves += 1
            return s.utility, None

        maximizing = s.current_player == "A"
        best = -inf if maximizing else inf
        best_action = None
        for action in _actions(s, order):
            next_state, _ = take_treasure(s, action, distances)
            value, _ = solve(next_state, remaining_depth - 1, current_depth + 1)
            if best_action is None or (maximizing and value > best) or (
                not maximizing and value < best
            ):
                best, best_action = value, action
        return int(best), best_action

    started = perf_counter()
    value, action = solve(state, depth, 0)
    elapsed_ms = (perf_counter() - started) * 1000
    return GameSearchResult(action, value, visited, leaves, 0, 0, elapsed_ms,
                            max_depth_reached=max_depth)


def alpha_beta_search(state: GameState, distances: DistanceMatrix,
                      depth: int, order: str = "left_to_right") -> GameSearchResult:
    """A-B剪枝；访问顺序默认T1→T2→T3→T4。"""
    if not (isinstance(depth, int) and 0 <= depth <= 4):
        raise ValueError("本题搜索深度必须介于0和4之间")
    _actions(state, order)
    visited = leaves = cutoffs = skipped = max_depth = 0

    def solve(s: GameState, remaining_depth: int,
              alpha: float, beta: float, current_depth: int) -> Tuple[int, Optional[str]]:
        nonlocal visited, leaves, cutoffs, skipped, max_depth
        visited += 1
        max_depth = max(max_depth, current_depth)
        if remaining_depth == 0 or not s.remaining:
            leaves += 1
            return s.utility, None

        maximizing = s.current_player == "A"
        best = -inf if maximizing else inf
        best_action = None
        actions = _actions(s, order)
        for i, action in enumerate(actions):
            next_state, _ = take_treasure(s, action, distances)
            value, _ = solve(next_state, remaining_depth - 1, alpha, beta, current_depth + 1)
            if best_action is None or (maximizing and value > best) or (
                not maximizing and value < best
            ):
                best, best_action = value, action

            if maximizing:
                alpha = max(alpha, best)
            else:
                beta = min(beta, best)

            if alpha >= beta:
                remaining_actions = len(actions) - i - 1
                if remaining_actions:
                    cutoffs += 1
                    skipped += remaining_actions
                break
        return int(best), best_action

    started = perf_counter()
    value, action = solve(state, depth, -inf, inf, 0)
    elapsed_ms = (perf_counter() - started) * 1000
    return GameSearchResult(action, value, visited, leaves, cutoffs,
                            skipped, elapsed_ms, max_depth_reached=max_depth)


def optimal_trace(state: GameState, distances: DistanceMatrix, depth: int):
    """用完整Minimax逐回合复原最优应对，供解释得分，不计入性能统计。"""
    steps = []
    for remaining_depth in range(depth, 0, -1):
        if not state.remaining:
            break
        result = minimax_search(state, distances, remaining_depth)
        target = result.best_target
        next_state, gain = take_treasure(state, target, distances)
        steps.append((state.current_player, state.pos_a if state.current_player == "A"
                      else state.pos_b, target, gain, next_state.score_a,
                      next_state.score_b, next_state.utility))
        state = next_state
    return steps


def run_tests() -> None:
    print("=== 实验二 / 题目四 第2阶段：Minimax与Alpha-Beta宝藏决策 ===")
    distances, _ = build_distance_matrix()  # 从第1阶段的A*算法复用已验证的矩阵计算
    state = initial_state()
    assert state.remaining == ("T1", "T2", "T3", "T4")
    assert state.utility == 0

    # 独立的状态转移检查：A选择T3的净收益是7-8=-1。
    after_a, gain = take_treasure(state, "T3", distances)
    assert gain == -1 and after_a.pos_a == "T3"
    assert after_a.pos_b == "B" and after_a.score_a == -1
    assert after_a.score_b == 0 and after_a.current_player == "B"
    assert after_a.remaining == ("T1", "T2", "T4")
    print("[PASS] 状态转移：A取T3，收益=-1，双方位置/剩余宝藏/回合正确更新")
    try:
        take_treasure(after_a, "T3", distances)
    except ValueError:
        print("[PASS] 已取得宝藏不能重复选取")
    else:
        raise AssertionError("同一个宝藏被重复选取")

    for depth, expected_target, expected_value in (
        (2, "T2", -1),
        (4, "T3", 3),
    ):
        mm = minimax_search(state, distances, depth)
        ab = alpha_beta_search(state, distances, depth)
        print(f"\n[H={depth}]")
        print(f"  Minimax    : best_target={mm.best_target}, best_value={mm.best_value}, "
              f"访问状态={mm.visited_states}, 叶子={mm.evaluated_leaves}")
        print(f"  Alpha-Beta : best_target={ab.best_target}, best_value={ab.best_value}, "
              f"访问状态={ab.visited_states}, 叶子={ab.evaluated_leaves}, "
              f"剪枝事件={ab.cutoff_events}, 跳过动作={ab.skipped_actions}")
        assert mm.best_target == ab.best_target == expected_target
        assert mm.best_value == ab.best_value == expected_value
        assert ab.visited_states <= mm.visited_states
        assert ab.evaluated_leaves <= mm.evaluated_leaves
        print(f"[PASS] PDF H={depth} 的最优首步与收益差一致：{expected_target}, {expected_value}")

        trace = optimal_trace(state, distances, depth)
        print("  完整Minimax对应的最优应对过程：")
        for turn, (player, old_pos, target, gain, score_a, score_b, utility) in enumerate(trace, 1):
            print(f"    回合{turn} {player}: {old_pos}->{target}, 本回合收益={gain:+d}, "
                  f"Score_A={score_a}, Score_B={score_b}, U={utility}")
        assert trace[-1][-1] == expected_value

    # H=0立即评估；终止时尚未取完的宝藏不产生额外收益。
    assert minimax_search(state, distances, 0).best_value == 0
    assert alpha_beta_search(state, distances, 0).best_target is None
    assert minimax_search(state, distances, 0).max_depth_reached == 0
    assert alpha_beta_search(state, distances, 0).max_depth_reached == 0
    print("\n[PASS] H=0时直接返回当前收益差")
    print("=== 第四题第2阶段基础功能检查全部通过 ===")


if __name__ == "__main__":
    run_tests()
