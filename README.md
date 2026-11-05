

# 当代人工智能实验二：搜索算法

本项目完成《当代人工智能实验课》实验二，内容包括：

- 题目一：单位代价迷宫上的 A* 搜索；
- 题目二：加权地形上的 A* 搜索；
- 题目三：Minimax 与 Alpha-Beta 剪枝；
- 题目四：A* 距离矩阵与宝藏分配博弈；
- 搜索效率、剪枝效率和访问顺序对照；
- 基于即时净收益的 Alpha-Beta 动作排序拓展。

所有课程给定结果均经过自动检查。实验数据由程序真实运行产生，不使用手工填写的节点数或计时结果。

## 1. 环境与依赖

本项目实际测试环境：

- 操作系统：Windows
- Python：3.9.12
- NumPy：2.0.2
- Matplotlib：3.9.4

核心搜索算法、测试和 CSV 实验仅使用 Python 标准库。

NumPy 和 Matplotlib 只用于生成报告图表。需要重新绘图时，可安装：

```powershell
python -m pip install numpy matplotlib
```

可以通过以下命令核对当前环境：

```powershell
python --version
python -c "import numpy; print(numpy.__version__)"
python -c "import matplotlib; print(matplotlib.__version__)"
```

## 2. 项目结构

```text
data_Project2/
├── .gitignore
├── README.md
├── requirements.txt
├── run_all_checks.py
├── 实验二要求_new.pdf
├── tests/
│   └── test_lab2.py
└── src/
    ├── __init__.py
    ├── grid_search.py
    ├── weighted_astar.py
    ├── game_tree.py
    ├── treasure_distance.py
    ├── treasure_game.py
    ├── experiment_q1.py
    ├── experiment_q2.py
    ├── experiment_q3.py
    ├── experiment_q4.py
    ├── experiment_innovation.py
    ├── experiment_precomputation.py
    ├── report_figures.py
    ├── results/
    └── figures/
```

各文件作用如下：

| 文件                         | 作用                               |
| ---------------------------- | ---------------------------------- |
| `requirements.txt`          | 绘图依赖及已验证版本               |
| `run_all_checks.py`         | 一键运行测试、实验和图表生成       |
| `src/__init__.py`           | 使核心模块支持按 `src` 包导入      |
| `grid_search.py`           | 题目一单位代价迷宫 A*              |
| `weighted_astar.py`        | 题目二加权地形 A*                  |
| `game_tree.py`             | 题目三 Minimax 与 Alpha-Beta       |
| `treasure_distance.py`     | 题目四 6×6 距离矩阵               |
| `treasure_game.py`         | 题目四宝藏博弈状态与搜索           |
| `experiment_q1.py`         | 题目一启发函数对照和 BFS 验证      |
| `experiment_q2.py`         | 题目二三种启发函数和 Dijkstra 验证 |
| `experiment_q3.py`         | Minimax、Alpha-Beta 和访问顺序对照 |
| `experiment_q4.py`         | 搜索深度、访问顺序和首步效用实验   |
| `experiment_innovation.py` | 即时净收益动作排序拓展             |
| `experiment_precomputation.py` | 距离矩阵预计算与按需A*对照     |
| `report_figures.py`        | 根据 CSV 生成报告图表              |
| `tests/test_lab2.py`       | 边界、异常、深度和反例测试         |

## 3. 运行方法

以下命令均在项目根目录运行。

### 3.1 一键最终验收

```powershell
python run_all_checks.py
```

该命令依次运行单元测试、四道题基础检查、对照实验、创新实验和图表生成。任一步失败会立即停止并返回非零退出码。

注意：一键验收会覆盖对应的 CSV 和 PNG/SVG；如果只想检查代码而不刷新实验数据，请仅运行下一节的单元测试。

### 3.2 单元测试

```powershell
python -m unittest discover -s tests -v
```

当前共有 31 个测试，覆盖：

- 课程给定迷宫；
- 起点等于终点；
- 无解地图；
- 唯一通路；
- 多条等长最短路径；
- 边界起终点；
- 非法坐标；
- 不规则和空地图；
- 起点或终点位于障碍物；
- 非法网格字符；
- 布尔坐标和非法加权地形；
- 非法启发函数名称；
- 普通曼哈顿次优反例；
- 加权地图给定样例、免费终点与无解返回；
- Minimax/Alpha-Beta访问顺序、剪枝和非法博弈树；
- 以 `src` 包方式导入全部算法模块；
- Minimax 和 Alpha-Beta 最大递归深度；
- H=2、H=4 结果；
- 完整6×6距离矩阵和宝藏状态转移；
- 不可达、无穷和 NaN 距离。

当前测试结果：

```text
Ran 31 tests
OK
```

### 3.3 四道题基础检查

```powershell
python src\grid_search.py
python src\weighted_astar.py
python src\game_tree.py
python src\treasure_distance.py
python src\treasure_game.py
```

### 3.4 对照与创新实验

```powershell
python src\experiment_q1.py
python src\experiment_q2.py
python src\experiment_q3.py
python src\experiment_q4.py
python src\experiment_innovation.py
python src\experiment_precomputation.py
```

### 3.5 重新生成图表

```powershell
python src\report_figures.py
```

重新运行实验会覆盖对应 CSV。不同机器和不同系统负载下，计时数据可能变化；节点数量、最优值和课程给定结果应保持一致。

## 4. 题目一：单位代价迷宫 A*

输入：

- 起点坐标 `Tuple[int, int]`；
- 终点坐标 `Tuple[int, int]`；
- 二维字符地图。

规则：

- 只能上下左右移动；
- `X` 为不可通过的障碍物；
- 每步代价为 1；
- 有解时返回从起点到终点的完整坐标序列；
- 无解时返回空列表。

A* 使用：

\[
f(n)=g(n)+h(n)
\]

其中：

- `g(n)` 是起点到当前节点的真实代价；
- `h(n)` 是当前节点到终点的估计代价；
- `f(n)` 是估计总代价。

题目一比较：

1. `h=0`，即一致代价搜索；
2. 曼哈顿距离。

两种方法都保证返回最短路径。曼哈顿距离可能减少展开节点，但效果也受到地图结构和相同 `f` 值节点处理顺序的影响。

课程样例结果：

- 路径代价：7；
- 解深度：7；
- 返回路径合法且为最短路径。

## 5. 题目二：加权地形 A*

移动代价等于进入目标格子的权值，进入终点 `E` 的代价为 0。

课程要求接口返回：

- 有解：最小总代价；
- 无解：`-1`。

程序内部同时保留路径和搜索统计，便于核对每一步累计代价。

本项目比较三种启发函数。

| 模式                   | 定义               | 最优性保证 |
| ---------------------- | ------------------ | ---------- |
| `zero`               | \(h(n)=0\)         | 有         |
| `manhattan`          | 原始曼哈顿距离     | 无         |
| `adjusted_manhattan` | 修正后的曼哈顿下界 | 有         |

修正启发函数为：

\[
h(n)=\max(D(n,E)-1,0)\times c_{\min}
\]

其中：

- \(D(n,E)\) 是曼哈顿距离；
- 最后进入 E 的一步免费，因此减去 1；
- \(c_{\min}\) 是非终点可通行格子的最低进入代价。

当地图包含零权格时，\(c_{\min}=0\)，修正启发函数会退化为 `h=0`，但仍保持最优性。

课程 5×5 样例中：

- 起点曼哈顿距离为 8；
- 真实最优代价为 7；
- 因此普通曼哈顿已经高估，不能保证最优；
- 三种方法在课程样例中都碰巧返回 7。

项目还包含确定性反例：

```python
[
    ["S", 1, 1, 1, "E"],
    [0,   0, 0, 0,  0],
]
```

结果为：

| 方法       | 返回代价 | 是否最优 |
| ---------- | -------: | -------- |
| `zero`   |        0 | 是       |
| 普通曼哈顿 |        3 | 否       |
| 修正曼哈顿 |        0 | 是       |

在固定种子生成的 200 组随机地图中：

- `zero`：200/200 与独立 Dijkstra 一致；
- 修正曼哈顿：200/200 与独立 Dijkstra 一致；
- 普通曼哈顿：199/200 返回最优结果。

这说明普通曼哈顿可能减少搜索量，但不能提供最优性保证。

## 6. 题目三：Minimax 与 Alpha-Beta

树节点采用课程给定结构：

- 叶节点包含 `value`；
- 内部节点包含 `children`；
- 根节点为 MAX；
- 默认从左到右访问子节点。

实现包括：

- 普通 Minimax；
- Alpha-Beta 剪枝；
- 从左到右访问；
- 从右到左访问；
- 离线理想顺序控制实验；
- 不利访问顺序控制实验。

课程给定样例结果：

```text
Minimax = 5
Alpha-Beta = 5
```

Alpha-Beta 中：

- `alpha` 表示 MAX 当前已经能够保证的下界；
- `beta` 表示 MIN 当前已经能够保证的上界；
- 当 `alpha >= beta` 时，后续兄弟子树不能改变祖先的最终决策，因此可以剪枝。

改变访问顺序不会改变最终 Minimax 值，但会改变剪枝数量和搜索开销。

离线理想顺序需要提前知道真实子树值，只用于展示访问顺序的理论影响，不能视为零成本的实用优化。

## 7. 题目四：宝藏分配

关键位置顺序固定为：

```text
A, B, T1, T2, T3, T4
```

6×6 最小移动代价矩阵为：

| from/to |  A |  B | T1 | T2 | T3 | T4 |
| ------- | -: | -: | -: | -: | -: | -: |
| A       |  0 | 12 | 10 |  4 |  8 |  8 |
| B       | 12 |  0 |  8 |  8 |  4 | 10 |
| T1      | 10 |  8 |  0 |  6 |  6 | 12 |
| T2      |  4 |  8 |  6 |  0 |  6 |  6 |
| T3      |  8 |  4 |  6 |  6 |  0 |  6 |
| T4      |  8 | 10 | 12 |  6 |  6 |  0 |

本题六个特殊位置的进入代价均为 1，因此矩阵对称。一般节点加权地图不一定满足距离对称。

博弈状态记录：

- A 的当前位置；
- B 的当前位置；
- 剩余宝藏；
- A 的累计收益；
- B 的累计收益；
- 当前玩家。

搜索深度通过递归参数 `remaining_depth` 传递。它属于搜索控制参数，不属于游戏局面本身，因此没有重复存入 `GameState`。已经执行的回合数可以由剩余宝藏数量推导。

评价函数为：

\[
U=Score_A-Score_B
\]

课程必测结果：

| 深度 | 最优首步 | 最优收益差 |
| ---: | -------- | ---------: |
|  H=2 | T2       |         -1 |
|  H=4 | T3       |          3 |

Minimax 和 Alpha-Beta 在相同深度与相同访问顺序下返回相同决策和收益值。

H=2 主要评价近期收益；H=4 会考虑完整四回合中对手的后续选择，因此最优首步从 T2 变为 T3。

## 8. 不可达位置策略

课程固定地图中六个关键位置全部可达。

固定距离矩阵构造时，如果发现关键位置不可达，程序会抛出异常，不会静默使用错误距离。

`take_treasure` 会拒绝：

- 负距离；
- 正无穷；
- 负无穷；
- NaN；
- 缺失距离。

如果扩展到一般地图，可以使用 `-1` 或 `math.inf` 表示不可达，在动作生成阶段过滤不可达宝藏；如果没有合法动作，则以当前 utility 结束搜索。

## 9. 搜索统计定义

### A* 指标

- `expanded_nodes`：正式取出并处理的节点数；
- `generated_nodes`：加入优先队列的条目数，包含起点；
- `stale_pops`：被跳过的失效队列项；
- `max_frontier_size`：优先队列最大条目数；
- `path_cost`：路径总代价；
- `solution_depth`：移动步数；
- `elapsed_seconds`：搜索执行时间。

### 博弈搜索指标

- `visited_nodes` 或 `visited_states`：实际进入递归的节点数；
- `evaluated_leaves`：实际执行评价的叶节点数；
- `cutoff_events`：发生有效剪枝的次数；
- `skipped_subtrees` 或 `skipped_actions`：直接跳过的子树或动作数；
- `pruned_nodes`：相对完整 Minimax 少访问的节点数；
- `max_depth_reached`：实际进入递归的最大深度，根节点深度为 0；
- `elapsed_ms`：搜索执行时间。

运行时间使用 120 次或 200 次重复实验的中位数。毫秒级计时容易受到系统负载影响，因此性能分析同时结合确定性的节点统计。

## 10. 创新实验

创新实验在 Alpha-Beta 中加入即时净收益动作排序。

每个博弈节点按照：

\[
Gain(T)=V(T)-d(p,T)
\]

对当前可选宝藏降序排序，同值时保持原顺序。

排序只改变访问顺序，不改变：

- 状态转移；
- 评价函数；
- 搜索深度；
- 最终 Minimax 值。

在 H=4 的课程综合题中：

- 固定左到右 Alpha-Beta 访问 47 个状态；
- 即时净收益排序访问 37 个状态；
- 减少约 21.3%；
- 最优首步仍为 T3；
- 最优收益差仍为 3。

排序开销计入运行时间。该方法不能保证在所有规模和场景中都更快。

### 10.1 距离矩阵预计算对照

`experiment_precomputation.py` 使用完全相同的 Minimax 规则比较：

1. 先用 A* 计算30个有序关键点对，搜索时直接查距离矩阵；
2. 每次展开宝藏动作时重新运行 A*，且不使用缓存。

实验分别测试单独 H=2、单独 H=4，以及连续运行 H=2+H=4 的复用场景。两种策略必须返回相同的首步与收益差，同时记录 A* 调用次数、累计展开/生成节点以及总耗时中位数。

这个对照用于分析预计算的适用条件：单次浅层搜索可能不足以抵消30次预计算的固定成本；搜索更深或重复运行多个深度时，距离矩阵可以复用，从而避免重复A*。实际结论必须以运行后生成的CSV为准。

## 11. 实验数据与图表

实验脚本生成以下 CSV：

```text
src/results/q1_astar_comparison.csv
src/results/q2_weighted_astar_comparison.csv
src/results/q3_minimax_alpha_beta_comparison.csv
src/results/q4_distance_matrix.csv
src/results/q4_pairwise_astar_stats.csv
src/results/q4_game_search_comparison.csv
src/results/q4_first_action_values.csv
src/results/q4_gain_ordering_comparison.csv
src/results/q4_precomputation_comparison.csv
```

图表位于：

```text
src/figures/
```

包括：

- 题目一启发函数节点数对照；
- 题目二三种启发函数对照；
- 题目三访问顺序与剪枝比例；
- 题目四搜索深度与访问状态数；
- H=2、H=4 首步效用；
- 动态排序真实时间对照。

报告中的所有数字必须与最终保留的 CSV 一致。如果重新运行实验并覆盖 CSV，应重新生成图表并同步修改报告。

## 12. 实验报告

最终报告应包含：

1. 实验目标与整体设置；
2. A* 的状态、`g/h/f`、优先队列和路径恢复；
3. 题目一 `h=0` 与曼哈顿距离对照；
4. 题目二路径代价和启发函数可采纳性；
5. Minimax 数值回传和 Alpha-Beta 剪枝；
6. 不同访问顺序的剪枝效果；
7. 题目四距离矩阵和状态设计；
8. H=2 与 H=4 的决策差异；
9. 动态动作排序创新；
10. 局限与总结。

报告页数、文件命名和提交方式以课程群最新通知为准。

报告尚未生成时，不应在 README 中声称报告 PDF 已存在。

## 13. 已知限制

- 普通曼哈顿在加权地图中不保证最优；
- 零权地形会使修正启发函数退化为 `h=0`；
- 当前题目四固定实现只处理四个给定宝藏；
- 离线理想子节点顺序需要真实子树值，不是实际免费优化；
- 小规模搜索的计时容易受到操作系统调度和系统负载影响；
- 一般不可达地图需要在动作生成阶段过滤不可达宝藏。

## 14. 提交前检查

- [ ] 四道题的基础程序全部通过；
- [ ] 20 个单元测试全部通过；
- [ ] 9 份 CSV 已由最终版本代码生成；
- [ ] 6 张图已根据最终 CSV 重新生成；
- [ ] 报告中的数字与 CSV 一致；
- [ ] 姓名和学号已填写；
- [ ] 报告格式符合课程群最新通知；
- [ ] README 中的环境版本与提交机器一致；
- [ ] 删除 `__pycache__/` 和 `.pyc`；
- [ ] 检查是否需要提交运行日志；
- [ ] 按助教最终通知打包和命名。
