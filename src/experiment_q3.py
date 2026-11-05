"""当代人工智能实验二 / 题目三：Minimax 与 Alpha-Beta 对照实验。

运行：python experiment_q3.py
依赖：同一目录下的 game_tree.py（不需要安装第三方库）

对照维度：
  1. 算法：Minimax、Alpha-Beta
  2. 子节点访问顺序：从左到右、从右到左
  3. 测试树：PDF 样例、固定随机树、离线理想排序的控制树

提示：第三组“理想排序树”使用已知完整子树的 Minimax 值对孩子进行事先排序，
仅用于展示访问顺序对剪枝效率的理论影响，不能算实际可用的在线启发式优化。
"""

import csv
from pathlib import Path
from random import Random
from statistics import median
from typing import List, Tuple

if __package__:
    from .game_tree import (
        TreeNode,
        alpha_beta_search,
        build_pdf_example,
        minimax_search,
    )
else:
    from game_tree import (
        TreeNode,
        alpha_beta_search,
        build_pdf_example,
        minimax_search,
    )

REPEATS = 120  # 每种算法×顺序：120次计时，中位数更稳健
WARMUPS = 5
BASE_DIR = Path(__file__).resolve().parent
OUTPUT = BASE_DIR / "results" / "q3_minimax_alpha_beta_comparison.csv"


def make_random_tree(branching: int, depth: int, seed: int) -> TreeNode:
    """固定随机种子生成满 b 叉博弈树；depth 是根到叶子的边数。"""
    if branching < 2 or depth < 1:
        raise ValueError("branching >= 2 且 depth >= 1")
    rng = Random(seed)

    def build(remaining: int) -> TreeNode:
        if remaining == 0:
            return TreeNode(value=rng.randint(-50, 50))
        return TreeNode(children=[build(remaining - 1) for _ in range(branching)])

    return build(depth)


def offline_ideal_order(root: TreeNode) -> None:
    """教学控制变量：按照真实子树 Minimax 值离线重排，同树上的L2R最优优先。

    仅用于研究节点顺序影响，排序前已计算整棵树，不计入搜索耗时。
    禁止在报告中把该控制实验描述为无需额外开销的实用排序优化。
    """
    def arrange(node: TreeNode, is_max: bool) -> float:
        if not node.children:
            return node.value
        child_scores = [(arrange(child, not is_max), child)
                        for child in node.children]
        child_scores.sort(key=lambda entry: entry[0], reverse=is_max)
        node.children[:] = [child for _, child in child_scores]
        return child_scores[0][0]

    arrange(root, True)


def check_random_consistency(tests: int = 200) -> None:
    """无随机外部依赖的正确性回归：两算法×两访问顺序×两种根角色。"""
    master_rng = Random(20261031)
    for _ in range(tests):
        depth = master_rng.randint(1, 4)
        branching = master_rng.randint(2, 3)
        seed = master_rng.randrange(10**9)
        root = make_random_tree(branching, depth, seed)
        for root_is_max in (True, False):
            values = []
            for order in ("left_to_right", "right_to_left"):
                mm = minimax_search(root, root_is_max=root_is_max, order=order)
                ab = alpha_beta_search(root, root_is_max=root_is_max, order=order)
                assert mm.best_value == ab.best_value, (depth, branching, seed)
                assert mm.visited_nodes == mm.total_nodes
                assert mm.evaluated_leaves == mm.total_leaves
                assert 1 <= ab.visited_nodes <= mm.visited_nodes
                assert 1 <= ab.evaluated_leaves <= mm.evaluated_leaves
                assert ab.pruned_nodes == ab.total_nodes - ab.visited_nodes
                values.append(mm.best_value)
            assert values[0] == values[1], "访问顺序不应影响最优值"
    print(f"[PASS] {tests}组固定随机树，双根角色、双顺序：Minimax与Alpha-Beta最优值一致")


def run_one(tree: TreeNode, method: str, order: str) -> Tuple[dict, object]:
    search = minimax_search if method == "Minimax" else alpha_beta_search
    for _ in range(WARMUPS):
        search(tree, order=order)
    results = [search(tree, order=order) for _ in range(REPEATS)]
    baseline = results[0]
    assert all((r.best_value, r.visited_nodes, r.evaluated_leaves, r.cutoff_events, r.max_depth_reached)
               == (baseline.best_value, baseline.visited_nodes,
                   baseline.evaluated_leaves, baseline.cutoff_events, baseline.max_depth_reached)
               for r in results), "重复运行结果不稳定"

    saving = (baseline.pruned_nodes / baseline.total_nodes) * 100
    row = {
        "scenario": "",  # 在主函数中填入
        "method": method,
        "visit_order": order,
        "best_value": baseline.best_value,
        "total_nodes": baseline.total_nodes,
        "visited_nodes": baseline.visited_nodes,
        "max_depth_reached": baseline.max_depth_reached,
        "pruned_nodes": baseline.pruned_nodes,
        "saved_node_pct": round(saving, 2),
        "total_leaves": baseline.total_leaves,
        "evaluated_leaves": baseline.evaluated_leaves,
        "pruned_leaves": baseline.pruned_leaves,
        "cutoff_events": baseline.cutoff_events,
        "skipped_subtrees": baseline.skipped_subtrees,
        "median_time_ms": round(median(r.elapsed_ms for r in results), 6),
        "repeat_count": REPEATS,
    }
    return row, baseline


def main() -> None:
    print("=== 实验二 / 题目三：Minimax vs Alpha-Beta 性能对照 ===")
    print(f"每种方法运行 {REPEATS} 次，报告搜索耗时中位数（ms）。")
    print("时间不含事先的树结构检查/规模统计，也不含离线重排成本。\n")

    pdf = build_pdf_example()
    random_tree = make_random_tree(branching=4, depth=4, seed=2026)
    ideal_tree = make_random_tree(branching=4, depth=5, seed=2026)
    offline_ideal_order(ideal_tree)

    cases = [
        ("PDF_example", pdf),
        ("RANDOM_4ARY_DEPTH4_SEED2026", random_tree),
        ("IDEAL_ORDER_4ARY_DEPTH5_SEED2026", ideal_tree),
    ]
    rows: List[dict] = []

    for name, tree in cases:
        print(f"[{name}]")
        case_results = {}
        for order in ("left_to_right", "right_to_left"):
            for method in ("Minimax", "Alpha-Beta"):
                row, result = run_one(tree, method, order)
                row["scenario"] = name
                rows.append(row)
                case_results[(method, order)] = result
                print(
                    f"  {method:10s} {order:13s} "
                    f"最优值={result.best_value:>3g} "
                    f"访问={result.visited_nodes:>4}/{result.total_nodes} "
                    f"叶子={result.evaluated_leaves:>4}/{result.total_leaves} "
                    f"剪枝事件={result.cutoff_events:>3} "
                    f"耗时中位数={row['median_time_ms']:.4f}ms"
                )

        base = case_results[("Minimax", "left_to_right")].best_value
        assert all(result.best_value == base for result in case_results.values())
        for order in ("left_to_right", "right_to_left"):
            mm = case_results[("Minimax", order)]
            ab = case_results[("Alpha-Beta", order)]
            # 每棵测试树的所有根叶路径等长，剪枝不能跳过所有最深层叶子。
            assert mm.max_depth_reached == ab.max_depth_reached
            ratio = ab.pruned_nodes / mm.total_nodes * 100
            print(f"  {order}：Alpha-Beta 减少节点访问 {ratio:.1f}%")
        if name == "PDF_example":
            assert base == 5
            assert case_results[("Minimax", "left_to_right")].visited_nodes == 15
            assert case_results[("Alpha-Beta", "left_to_right")].visited_nodes == 14
        print("  [PASS] 四种配置最佳得分一致\n")

    print("=== 扩展正确性回归 ===")
    check_random_consistency(tests=200)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    assert len(rows) == 3 * 2 * 2
    print(f"\n[PASS] 已输出 {len(rows)} 行对照数据：{OUTPUT}")
    print("[提醒] IDEAL_ORDER为离线已知真实子树得分的控制条件，不能宣称实用算法无成本取得此顺序。")
    print("=== 题目三：性能对照与正确性回归全部通过 ===")


if __name__ == "__main__":
    main()
