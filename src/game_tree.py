from dataclasses import dataclass
from math import inf, isfinite
from time import perf_counter
from typing import List, Optional, Tuple


class TreeNode:
    """叶子有 value，内部节点有 children。"""

    def __init__(self, value: Optional[float] = None,
                 children: Optional[List["TreeNode"]] = None):
        self.value = value
        self.children = children if children is not None else []


@dataclass
class SearchResult:
    best_value: float
    visited_nodes: int          # 实际进入递归的节点数（包含内部节点和叶子）
    evaluated_leaves: int      # 实际计算的叶子数
    total_nodes: int           # 完整展开树的理论节点数
    total_leaves: int          # 完整展开树的理论叶子数
    cutoff_events: int        # 真正跳过后续兄弟子树的剪枝发生次数
    skipped_subtrees: int      # 直接跳过的兄弟子树数量
    elapsed_ms: float          # 算法搜索耗时（不含输入校验和树规模统计）
    max_depth_reached: int = 0 # 根深度0；仅统计实际访问的最大递归深度

    @property
    def pruned_nodes(self) -> int:
        """相对完整 Minimax，少进入递归的节点总数。"""
        return self.total_nodes - self.visited_nodes

    @property
    def pruned_leaves(self) -> int:
        return self.total_leaves - self.evaluated_leaves


def _tree_size(node: TreeNode, ancestors: set) -> Tuple[int, int]:
    """检验结构并计算节点、叶子总数。允许样例中子树被共享，但不允许环。"""
    if not isinstance(node, TreeNode):
        raise TypeError("树节点必须是 TreeNode")
    ident = id(node)
    if ident in ancestors:
        raise ValueError("检测到循环引用：博弈树不能包含环")
    if not isinstance(node.children, list):
        raise TypeError("children 必须是 list")

    if not node.children:
        if not isinstance(node.value, (int, float)) or not isfinite(node.value):
            raise ValueError("叶子节点必须提供有限的数值 value")
        return 1, 1

    ancestors.add(ident)
    nodes, leaves = 1, 0
    try:
        for child in node.children:
            cn, cl = _tree_size(child, ancestors)
            nodes += cn
            leaves += cl
    finally:
        ancestors.remove(ident)
    return nodes, leaves


def _children(node: TreeNode, order: str) -> List[TreeNode]:
    if order == "left_to_right":
        return node.children
    if order == "right_to_left":
        return list(reversed(node.children))
    raise ValueError("order 只能是 'left_to_right' 或 'right_to_left'")


def minimax_search(root: TreeNode, root_is_max: bool = True,
                   order: str = "left_to_right") -> SearchResult:
    """完整搜索；对照基线，不实施剪枝。"""
    if order not in ("left_to_right", "right_to_left"):
        raise ValueError("非法访问顺序")
    total_nodes, total_leaves = _tree_size(root, set())
    visited = leaves = max_depth = 0

    def solve(node: TreeNode, is_max: bool, current_depth: int) -> float:
        nonlocal visited, leaves, max_depth
        visited += 1
        max_depth = max(max_depth, current_depth)
        if not node.children:
            leaves += 1
            return node.value

        if is_max:
            best = -inf
            for child in _children(node, order):
                best = max(best, solve(child, False, current_depth + 1))
        else:
            best = inf
            for child in _children(node, order):
                best = min(best, solve(child, True, current_depth + 1))
        return best

    started = perf_counter()
    value = solve(root, root_is_max, 0)
    elapsed_ms = (perf_counter() - started) * 1000
    return SearchResult(value, visited, leaves, total_nodes, total_leaves,
                        cutoff_events=0, skipped_subtrees=0,
                        elapsed_ms=elapsed_ms, max_depth_reached=max_depth)


def alpha_beta_search(root: TreeNode, root_is_max: bool = True,
                      order: str = "left_to_right") -> SearchResult:
    """Alpha-Beta 剪枝，默认严格从左到右访问。"""
    if order not in ("left_to_right", "right_to_left"):
        raise ValueError("非法访问顺序")
    total_nodes, total_leaves = _tree_size(root, set())
    visited = leaves = cutoffs = skipped = max_depth = 0

    def solve(node: TreeNode, is_max: bool, alpha: float, beta: float,
              current_depth: int) -> float:
        nonlocal visited, leaves, cutoffs, skipped, max_depth
        visited += 1
        max_depth = max(max_depth, current_depth)
        if not node.children:
            leaves += 1
            return node.value

        children = _children(node, order)
        best = -inf if is_max else inf
        for index, child in enumerate(children):
            child_value = solve(child, not is_max, alpha, beta, current_depth + 1)
            if is_max:
                best = max(best, child_value)
                alpha = max(alpha, best)
            else:
                best = min(best, child_value)
                beta = min(beta, best)

            # alpha >= beta 时，后续子节点不可能改变当前祖先的最优决策。
            # 仅当存在未访问的兄弟子树时，才记录一次实际剪枝事件。
            if alpha >= beta:
                remaining = len(children) - index - 1
                if remaining:
                    cutoffs += 1
                    skipped += remaining
                break
        return best

    started = perf_counter()
    value = solve(root, root_is_max, -inf, inf, 0)
    elapsed_ms = (perf_counter() - started) * 1000
    return SearchResult(value, visited, leaves, total_nodes, total_leaves,
                        cutoff_events=cutoffs, skipped_subtrees=skipped,
                        elapsed_ms=elapsed_ms, max_depth_reached=max_depth)


def minimax(root: TreeNode) -> float:
    """返回根节点为 MAX 时的最佳得分。"""
    return minimax_search(root).best_value


def alpha_beta(root: TreeNode) -> float:
    """返回根节点为 MAX 时的最佳得分。"""
    return alpha_beta_search(root).best_value


def build_pdf_example() -> TreeNode:
    """特别保留 max2 的两次引用。"""
    leaf1 = TreeNode(value=3)
    leaf2 = TreeNode(value=5)
    leaf3 = TreeNode(value=6)
    leaf4 = TreeNode(value=9)
    leaf5 = TreeNode(value=1)
    leaf6 = TreeNode(value=2)

    max1 = TreeNode(children=[leaf1, leaf2])
    max2 = TreeNode(children=[leaf3, leaf4])
    max3 = TreeNode(children=[leaf5, leaf6])
    min1 = TreeNode(children=[max1, max2])
    min2 = TreeNode(children=[max2, max3])
    return TreeNode(children=[min1, min2])


def show_result(name: str, result: SearchResult) -> None:
    print(f"{name}: 最优得分={result.best_value:g}, "
          f"访问节点={result.visited_nodes}/{result.total_nodes}, "
          f"评估叶子={result.evaluated_leaves}/{result.total_leaves}, "
          f"剪枝事件={result.cutoff_events}, "
          f"跳过子树={result.skipped_subtrees}, "
          f"少访问节点={result.pruned_nodes}, 最大访问深度={result.max_depth_reached}, "
          f"耗时={result.elapsed_ms:.6f}ms")


def run_tests() -> None:
    print("=== 实验二 / 题目三：Minimax 与 Alpha-Beta ===")
    root = build_pdf_example()
    print("\n[PDF给定样例：根节点 MAX，子节点从左到右]")
    basic = minimax_search(root)
    pruned = alpha_beta_search(root)
    show_result("Minimax   ", basic)
    show_result("Alpha-Beta", pruned)

    assert minimax(root) == 5
    assert alpha_beta(root) == 5
    assert basic.best_value == pruned.best_value == 5
    assert basic.total_leaves == 8
    assert basic.visited_nodes == basic.total_nodes
    assert pruned.evaluated_leaves < basic.evaluated_leaves
    print("[PASS] PDF给定样例：两种算法均返回最佳得分 5")
    print("[PASS] Alpha-Beta 确实跳过了不影响最优结果的叶子")

    print("\n[不同访问顺序：只验证最优值一致；性能对照放在下一步]")
    for order in ("left_to_right", "right_to_left"):
        mm = minimax_search(root, order=order)
        ab = alpha_beta_search(root, order=order)
        assert mm.best_value == ab.best_value == 5
        print(f"[PASS] {order}: Minimax={mm.best_value:g}, "
              f"Alpha-Beta={ab.best_value:g}")

    print("\n[边界与异常输入]")
    assert alpha_beta(TreeNode(value=7)) == 7
    assert minimax(TreeNode(value=7)) == 7
    print("[PASS] 只有一个叶子的树")

    negative = TreeNode(children=[
        TreeNode(children=[TreeNode(value=-5), TreeNode(value=-1)]),
        TreeNode(children=[TreeNode(value=-3), TreeNode(value=-2)]),
    ])
    assert minimax(negative) == alpha_beta(negative) == -3
    print("[PASS] 负数叶子得分与 MIN/MAX 层次处理")

    try:
        alpha_beta(TreeNode())
    except ValueError:
        print("[PASS] 缺少叶子得分时正确报错")
    else:
        raise AssertionError("无效树未报错")

    print("\n=== 题目三：基础功能检查全部通过 ===")


if __name__ == "__main__":
    run_tests()
