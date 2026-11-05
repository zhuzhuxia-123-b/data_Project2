"""当代人工智能实验二：从已有CSV生成报告图表（不重复执行算法）。

用法（本文件放在 src/ 目录，与 results/ 平级）:
    python report_figures.py

依赖：matplotlib（没有则运行 python -m pip install matplotlib）
输入：results/q1_astar_comparison.csv 等五份CSV
输出：figures/ 下的六张 PNG 图和对应 SVG 矢量图

图表只读取本机的实验结果；不会使用作者机器上的计时值。
"""
from __future__ import annotations

import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # 不依赖GUI，适合VS Code终端
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# 统一为便于中文报告引用的清晰英文图注；不依赖Windows专有中文字体
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5,
    "axes.titlesize": 13, "axes.labelsize": 10.5,
    "figure.dpi": 110, "savefig.dpi": 230,
    "axes.spines.top": False, "axes.spines.right": False,
})


def read_csv(filename: str, required: tuple[str, ...]) -> list[dict]:
    path = RESULTS / filename
    if not path.is_file():
        raise FileNotFoundError(f"找不到实验数据文件：{path}\n先确认已经运行前面各题的实验脚本。")
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{filename} 缺少字段：{', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{filename} 为空文件")
    return rows


def one(rows: list[dict], **conditions) -> dict:
    selected = [row for row in rows if all(str(row[k]) == str(v) for k, v in conditions.items())]
    if len(selected) != 1:
        raise ValueError(f"配置 {conditions} 应恰好对应一行，实际找到 {len(selected)} 行")
    return selected[0]


def cosmetics(ax, *, axis="y"):
    ax.grid(axis=axis, alpha=0.18)
    ax.set_axisbelow(True)


def save(fig, name: str):
    for ext in ("png", "svg"):
        fig.savefig(FIGURES / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] figures/{name}.png 和 .svg")


def make_q1():
    rows = read_csv("q1_astar_comparison.csv", ("case", "heuristic", "expanded_nodes", "optimal_cost"))
    cases = [
        ("PDF_5x5_maze", "PDF 5x5 maze"),
        ("OPEN_15x15_same_row", "15x15 same row"),
        ("OPEN_15x15_opposite_corners", "15x15 corners"),
    ]
    # 同一场景两种启发函数必须返回同一最优代价
    for key, _ in cases:
        assert one(rows, case=key, heuristic="zero")["optimal_cost"] == one(rows, case=key, heuristic="manhattan")["optimal_cost"]
    zero = [int(one(rows, case=k, heuristic="zero")["expanded_nodes"]) for k, _ in cases]
    man = [int(one(rows, case=k, heuristic="manhattan")["expanded_nodes"]) for k, _ in cases]
    x = np.arange(len(cases))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars1 = ax.bar(x - .19, zero, .37, label="h = 0 (Dijkstra)")
    bars2 = ax.bar(x + .19, man, .37, label="Manhattan")
    ax.bar_label(bars1, padding=3, fontsize=9)
    ax.bar_label(bars2, padding=3, fontsize=9)
    ax.set_xticks(x, [label for _, label in cases])
    ax.set_ylabel("Expanded nodes")
    ax.set_title("Q1 | A* heuristic comparison")
    ax.legend(frameon=False, ncol=2)
    ax.set_ylim(0, max(zero + man) * 1.17)
    cosmetics(ax)
    fig.tight_layout()
    save(fig, "fig01_q1_heuristic_nodes")


def make_q2():
    rows = read_csv("q2_weighted_astar_comparison.csv",
                    ("case", "heuristic", "expanded_nodes", "reference_optimal_cost",
                     "returned_cost", "is_optimal", "optimality_guaranteed"))
    cases = [
        ("PDF_5x5_weighted", "PDF 5x5"),
        ("FEWER_STEPS_HIGHER_COST", "High-cost shortcut"),
        ("OPEN_15x15_same_row", "15x15 same row"),
        ("MIXED_12x12_seed_2026", "12x12 mixed"),
        ("ZERO_WEIGHT_5x5", "Zero-weight 5x5"),
        ("ORDINARY_MANHATTAN_COUNTEREXAMPLE", "Counterexample 2x5"),
    ]
    modes = (
        ("zero", "h=0 (optimal)"),
        ("manhattan", "Ordinary Manhattan (NOT guaranteed)"),
        ("adjusted_manhattan", "Adjusted Manhattan (optimal)"),
    )
    for key, _ in cases:
        ref = one(rows, case=key, heuristic="zero")["reference_optimal_cost"]
        for mode in ("zero", "adjusted_manhattan"):
            row = one(rows, case=key, heuristic=mode)
            assert row["returned_cost"] == ref and row["optimality_guaranteed"] == "1"
        assert one(rows, case=key, heuristic="manhattan")["optimality_guaranteed"] == "0"
    y = np.arange(len(cases))
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    allvals = []
    offsets = (-.27, 0, .27)
    for (mode, label), offset in zip(modes, offsets):
        selected = [one(rows, case=k, heuristic=mode) for k, _ in cases]
        data = [int(r["expanded_nodes"]) for r in selected]
        allvals.extend(data)
        bars = ax.barh(y + offset, data, .25, label=label)
        ax.bar_label(bars, padding=3, fontsize=8)
    ax.set_yticks(y, [label for _, label in cases])
    ax.invert_yaxis()
    ax.set_xlim(0, max(allvals) * 1.27)
    ax.set_xlabel("Expanded nodes")
    ax.set_title("Q2 | Three heuristics: ordinary Manhattan does NOT guarantee optimality")
    ax.legend(frameon=False, ncol=1, loc="lower right", fontsize=9)
    counter = one(rows, case="ORDINARY_MANHATTAN_COUNTEREXAMPLE", heuristic="manhattan")
    assert counter["is_optimal"] == "0"
    fig.text(.53, .025, "2x5 counterexample: ordinary returns cost " +
             counter["returned_cost"] + " vs Dijkstra optimal " +
             counter["reference_optimal_cost"] + " (gap " + counter["optimality_gap"] + ")",
             ha="center", fontsize=10)
    cosmetics(ax, axis="x")
    fig.subplots_adjust(left=.23, bottom=.12, right=.98, top=.89)
    save(fig, "fig02_q2_weighted_nodes")


def make_q3():
    rows = read_csv("q3_minimax_alpha_beta_comparison.csv", ("scenario", "method", "visit_order", "visited_nodes", "total_nodes", "best_value"))
    scenarios = [
        ("PDF_example", "PDF example"),
        ("RANDOM_4ARY_DEPTH4_SEED2026", "4-ary depth 4 (random)"),
        ("IDEAL_ORDER_4ARY_DEPTH5_SEED2026", "4-ary depth 5 (oracle order)"),
    ]
    values = {"left_to_right": [], "right_to_left": []}
    for name, _ in scenarios:
        mm = one(rows, scenario=name, method="Minimax", visit_order="left_to_right")
        for order in values:
            ab = one(rows, scenario=name, method="Alpha-Beta", visit_order=order)
            assert ab["best_value"] == mm["best_value"]
            total = int(mm["total_nodes"])
            assert int(mm["visited_nodes"]) == total
            values[order].append(100 * (total - int(ab["visited_nodes"])) / total)
    x = np.arange(len(scenarios))
    fig, ax = plt.subplots(figsize=(10, 5))
    a = ax.bar(x - .19, values["left_to_right"], .37, label="Left to right")
    b = ax.bar(x + .19, values["right_to_left"], .37, label="Right to left")
    ax.bar_label(a, labels=[f"{v:.1f}%" for v in values["left_to_right"]], padding=3, fontsize=9)
    ax.bar_label(b, labels=[f"{v:.1f}%" for v in values["right_to_left"]], padding=3, fontsize=9)
    ax.set_xticks(x, [label for _, label in scenarios])
    ax.set_ylabel("Reduction in visited nodes vs Minimax (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Q3 | Alpha-Beta pruning depends on visit order")
    ax.legend(frameon=False, ncol=2)
    cosmetics(ax)
    fig.text(.5, .015, "Oracle order uses offline exact subtree values; its ordering cost is excluded.", ha="center", fontsize=9)
    fig.subplots_adjust(bottom=.22)
    save(fig, "fig03_q3_pruning_by_order")


def make_q4():
    rows = read_csv("q4_game_search_comparison.csv", ("depth", "algorithm", "order", "visited_states", "best_target", "best_value"))
    innov = read_csv("q4_gain_ordering_comparison.csv", ("depth", "algorithm", "visited_states", "best_target", "best_value", "median_elapsed_ms"))
    depths = [1, 2, 3, 4]
    series = [
        ("Minimax", [int(one(rows, depth=h, algorithm="Minimax", order="left_to_right")["visited_states"]) for h in depths]),
        ("Alpha-Beta (fixed)", [int(one(rows, depth=h, algorithm="Alpha-Beta", order="left_to_right")["visited_states"]) for h in depths]),
        ("Alpha-Beta (gain ordered)", [int(one(innov, depth=h, algorithm="AlphaBeta_gain_desc")["visited_states"]) for h in depths]),
    ]
    for h in depths:
        original = one(rows, depth=h, algorithm="Minimax", order="left_to_right")
        optimized = one(innov, depth=h, algorithm="AlphaBeta_gain_desc")
        assert original["best_value"] == optimized["best_value"]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for label, points in series:
        ax.plot(depths, points, "o-", markersize=7, linewidth=2, label=label)
        # 只标注最右端数值，避免 H=1/H=2 多条曲线接近时文字重叠。
        ax.annotate(str(points[-1]), (4, points[-1]), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.annotate("5 (all)", (1, 5), xytext=(0, 9),
                textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(depths)
    ax.set_xlim(.7, 4.3)
    ax.set_ylim(0, 80)
    ax.set_xlabel("Search depth H")
    ax.set_ylabel("Visited game states")
    ax.set_title("Q4 | Game-tree search cost by depth")
    ax.legend(frameon=False, loc="upper left")
    cosmetics(ax)
    fig.tight_layout()
    save(fig, "fig04_q4_depth_states")

    # 全部候选首步的 Minimax 收益：突出 H=2 到 H=4 的决策变化
    action_rows = read_csv("q4_first_action_values.csv", ("depth", "first_target", "final_minimax_utility", "is_optimal"))
    targets = ["T1", "T2", "T3", "T4"]
    a2 = [int(one(action_rows, depth=2, first_target=t)["final_minimax_utility"]) for t in targets]
    a4 = [int(one(action_rows, depth=4, first_target=t)["final_minimax_utility"]) for t in targets]
    assert (targets[int(np.argmax(a2))], max(a2)) == ("T2", -1)
    assert (targets[int(np.argmax(a4))], max(a4)) == ("T3", 3)
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(8, 5))
    left = ax.bar(x - .19, a2, .37, label="H = 2")
    right = ax.bar(x + .19, a4, .37, label="H = 4")
    ax.bar_label(left, labels=[f"{v:+d}" for v in a2], padding=3)
    ax.bar_label(right, labels=[f"{v:+d}" for v in a4], padding=3)
    ax.set_xticks(x, targets)
    ax.axhline(0, color="gray", linewidth=.9)
    ax.set_ylim(min(a2 + a4) - 2.3, max(a2 + a4) + 2)
    ax.set_ylabel("Final utility U = Score A - Score B")
    ax.set_title("Q4 | Best first action changes with search depth")
    ax.legend(frameon=False, ncol=2)
    cosmetics(ax)
    fig.tight_layout()
    save(fig, "fig05_q4_first_action_utility")

    # 计时图：使用用户机器上导出的实测CSV，包含动态排序时间成本
    fixed_times = [float(one(innov, depth=h, algorithm="AlphaBeta_left_to_right")["median_elapsed_ms"]) for h in depths]
    sorted_times = [float(one(innov, depth=h, algorithm="AlphaBeta_gain_desc")["median_elapsed_ms"]) for h in depths]
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(8.5, 5.1))
    a = ax.bar(x - .19, fixed_times, .37, label="Fixed order")
    b = ax.bar(x + .19, sorted_times, .37, label="Gain ordered (includes sorting)")
    ax.bar_label(a, labels=[f"{v:.4f}" for v in fixed_times], padding=3, fontsize=9)
    ax.bar_label(b, labels=[f"{v:.4f}" for v in sorted_times], padding=3, fontsize=9)
    ax.set_xticks(x, [f"H = {h}" for h in depths])
    ax.set_ylabel("Median search time (ms; 120 runs)")
    ax.set_ylim(0, max(fixed_times + sorted_times) * 1.21)
    ax.set_title("Extension | Dynamic ordering: actual time trade-off")
    ax.legend(frameon=False)
    cosmetics(ax)
    fig.text(.5, .015, "Short timings fluctuate with system load; node counts are more stable.", ha="center", fontsize=9)
    fig.subplots_adjust(bottom=.18)
    save(fig, "fig06_innovation_median_time")


def main():
    print("=== 实验二：报告图表自动生成 ===")
    make_q1()
    make_q2()
    make_q3()
    make_q4()
    print("[PASS] 共6张图，PNG + SVG双格式，数据来自你本机 results/ 下的CSV")
    print(f"输出目录：{FIGURES}")


if __name__ == "__main__":
    main()
