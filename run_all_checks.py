"""一键运行实验二的测试、基础样例、对照实验、创新实验和图表生成。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

TASKS = (
    ("单元测试", ("-m", "unittest", "discover", "-s", "tests", "-v")),
    ("题目一基础检查", ("src/grid_search.py",)),
    ("题目二基础检查", ("src/weighted_astar.py",)),
    ("题目三基础检查", ("src/game_tree.py",)),
    ("题目四距离矩阵", ("src/treasure_distance.py",)),
    ("题目四博弈搜索", ("src/treasure_game.py",)),
    ("题目一对照实验", ("src/experiment_q1.py",)),
    ("题目二对照实验", ("src/experiment_q2.py",)),
    ("题目三对照实验", ("src/experiment_q3.py",)),
    ("题目四对照实验", ("src/experiment_q4.py",)),
    ("创新实验", ("src/experiment_innovation.py",)),
    ("距离矩阵预计算对照", ("src/experiment_precomputation.py",)),
    ("生成图表", ("src/report_figures.py",)),
)


def main() -> int:
    print("=== 实验二：一键最终验收 ===", flush=True)
    print(f"Python: {sys.version.split()[0]}", flush=True)
    print(f"项目目录: {ROOT}", flush=True)
    print("注意：实验脚本会覆盖对应CSV，作图脚本会覆盖已有PNG/SVG。", flush=True)

    completed = []
    for index, (name, arguments) in enumerate(TASKS, 1):
        command = [sys.executable, "-B", *arguments]
        print(f"\n[{index}/{len(TASKS)}] {name}", flush=True)
        print("运行:", " ".join(command), flush=True)
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode != 0:
            print(f"\n[FAIL] {name}，退出码={result.returncode}", flush=True)
            print("后续步骤未运行；请先修复本步骤。", flush=True)
            return result.returncode
        completed.append(name)
        print(f"[PASS] {name}", flush=True)

    print("\n=== 最终验收全部通过 ===", flush=True)
    print(f"完成 {len(completed)} 个步骤。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

