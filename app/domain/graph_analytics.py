"""交易对手关系图的团伙识别与中心度分析（AI 增强 S11+）。

在既有「交易对手关系基础指标（度数/笔数/金额）」之外，进一步从关系图结构中挖掘
**团伙（资金环路/关联群组）** 与 **节点重要性（资金枢纽）** 两类风控价值更高的指标，
对标「关联网络反欺诈」卖点：

- 团伙识别：以无向关系图的**连通分量**界定关联群组（互相直接/间接交易的交易对手构成一个
  群组）。为每个节点产出：
    - ``{prefix}_community_size``：所属群组的节点规模（群组内交易主体/对手总数）。
    - ``{prefix}_ring_flag``：是否处于「疑似团伙」（群组规模 ≥ 阈值，默认 3）→ 1.0/0.0。
      规则可直接引用，如 ``ai_cp_ring_flag == 1`` 触发人工复核。
- 中心度：以 **PageRank** 衡量节点在资金网络中的重要性（被越多重要节点指向越重要），
  产出 ``{prefix}_pagerank``。资金枢纽/集中度异常的节点分值更高，可用于识别「中介账户」。

设计取舍：
- **纯 Python、确定性**：连通分量用并查集、PageRank 用固定阻尼/迭代的幂迭代，相同输入
  产出完全相同结果，便于业务集成测试断言，且无需任何第三方依赖（与既有回退策略一致）。
- **独立于基础指标**：本模块不改动 `build_counterparty_metrics`（基础三指标），新指标经
  独立提取器产出、由组合提取器合并，避免破坏既有「基础指标与参考实现一致」的约束。

DDD 分层：本模块属 domain 层，仅依赖标准库与领域内既有值对象（CounterpartyMetric / 
CounterpartyTransaction / validate_ref_name）。
"""

from __future__ import annotations

from typing import Iterable

from app.domain.counterparty import (
    CounterpartyMetric,
    CounterpartyTransaction,
    DEFAULT_REF_NAME_PREFIX,
    validate_ref_name,
)

# 疑似团伙的最小群组规模（节点数 ≥ 该值视为团伙）
RING_SIZE_THRESHOLD_DEFAULT = 3

# PageRank 参数（固定以保证确定性）
_PAGERANK_DAMPING = 0.85
_PAGERANK_ITERATIONS = 100
_PAGERANK_TOLERANCE = 1e-9


def _collect_edges(
    transactions: Iterable[CounterpartyTransaction],
) -> tuple[dict[str, set[str]], list[str]]:
    """从交易关系边构建无向邻接表（忽略自环与空白主体），返回 (邻接表, 有序节点列表)。

    节点列表按字典序排序，保证后续计算与输出的确定性。
    """
    adjacency: dict[str, set[str]] = {}

    def _touch(node: str) -> None:
        adjacency.setdefault(node, set())

    for txn in transactions:
        source = (txn.source or "").strip()
        target = (txn.target or "").strip()
        if not source or not target:
            continue
        _touch(source)
        _touch(target)
        if source == target:
            # 自环不构成与他人的关联，仅保留节点本身
            continue
        adjacency[source].add(target)
        adjacency[target].add(source)

    nodes = sorted(adjacency.keys())
    return adjacency, nodes


def _connected_components(
    adjacency: dict[str, set[str]], nodes: list[str]
) -> dict[str, int]:
    """并查集求连通分量，返回 {节点: 所属分量的节点规模}。"""
    parent: dict[str, str] = {n: n for n in nodes}

    def find(x: str) -> str:
        root = x
        while parent[root] != root:
            root = parent[root]
        # 路径压缩
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # 按字典序选较小者为根，保证确定性
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    for node in nodes:
        for neighbor in adjacency[node]:
            union(node, neighbor)

    size_by_root: dict[str, int] = {}
    for node in nodes:
        root = find(node)
        size_by_root[root] = size_by_root.get(root, 0) + 1

    return {node: size_by_root[find(node)] for node in nodes}


def _pagerank(adjacency: dict[str, set[str]], nodes: list[str]) -> dict[str, float]:
    """无向图 PageRank（确定性幂迭代）。孤立节点保留基础均匀分值。"""
    n = len(nodes)
    if n == 0:
        return {}
    rank = {node: 1.0 / n for node in nodes}
    base = (1.0 - _PAGERANK_DAMPING) / n

    for _ in range(_PAGERANK_ITERATIONS):
        new_rank: dict[str, float] = {}
        dangling_sum = 0.0
        for node in nodes:
            if not adjacency[node]:
                dangling_sum += rank[node]
        dangling = _PAGERANK_DAMPING * dangling_sum / n

        for node in nodes:
            incoming = 0.0
            for neighbor in adjacency[node]:
                deg = len(adjacency[neighbor])
                if deg > 0:
                    incoming += rank[neighbor] / deg
            new_rank[node] = base + dangling + _PAGERANK_DAMPING * incoming

        # 归一化，防止数值漂移
        total = sum(new_rank.values()) or 1.0
        new_rank = {node: value / total for node, value in new_rank.items()}

        delta = sum(abs(new_rank[node] - rank[node]) for node in nodes)
        rank = new_rank
        if delta < _PAGERANK_TOLERANCE:
            break

    return rank


def _louvain_communities(
    adjacency: dict[str, set[str]], nodes: list[str]
) -> dict[str, int]:
    """确定性 Louvain 社区发现（模块度优化），返回 {节点: 社区序号}。

    相比「连通分量」，Louvain 能在一个大连通图内进一步切分出**内部连接紧密**的社区
    （团伙），更贴合「资金小圈子」语义。本实现为纯 Python 单趟局部移动 + 确定性遍历
    （节点按字典序、邻居按字典序），保证相同输入产出相同社区划分，无第三方依赖。

    社区序号在最后按「社区内最小节点的字典序」重新编号，使序号本身也确定。
    """
    # 无边图：每个节点自成一社区
    if not nodes:
        return {}

    # 初始：每个节点独立成社区
    community: dict[str, int] = {node: i for i, node in enumerate(nodes)}
    degree: dict[str, int] = {node: len(adjacency[node]) for node in nodes}
    total_degree = sum(degree.values())
    if total_degree == 0:
        # 无边：各自独立，规范化编号后返回
        return _renumber_communities(community, nodes)

    m2 = float(total_degree)  # 2m（无向图度数和 = 2*边数）

    # 社区的总度数（用于模块度增益计算）
    sigma_tot: dict[int, float] = {}
    for node in nodes:
        sigma_tot[community[node]] = sigma_tot.get(community[node], 0.0) + degree[node]

    improved = True
    max_passes = 20
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for node in nodes:  # 确定性顺序
            current_comm = community[node]
            ki = degree[node]

            # 统计 node 到各邻居社区的连边数
            neigh_comm_weight: dict[int, int] = {}
            for neighbor in sorted(adjacency[node]):  # 确定性顺序
                c = community[neighbor]
                neigh_comm_weight[c] = neigh_comm_weight.get(c, 0) + 1

            # 先把 node 从当前社区移除
            sigma_tot[current_comm] -= ki

            # 选择模块度增益最大的目标社区（含留在原社区），并列时取社区序号最小者
            best_comm = current_comm
            best_gain = 0.0
            # 候选：所有邻居社区 + 原社区
            candidates = set(neigh_comm_weight.keys())
            candidates.add(current_comm)
            for c in sorted(candidates):
                k_in = neigh_comm_weight.get(c, 0)
                # 模块度增益（去掉常数项后用于比较）：k_in - sigma_tot[c]*ki/2m
                gain = k_in - sigma_tot.get(c, 0.0) * ki / m2
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_comm = c

            # 落定到 best_comm
            community[node] = best_comm
            sigma_tot[best_comm] = sigma_tot.get(best_comm, 0.0) + ki
            if best_comm != current_comm:
                improved = True

    return _renumber_communities(community, nodes)


def _renumber_communities(
    community: dict[str, int], nodes: list[str]
) -> dict[str, int]:
    """按「社区内最小节点字典序」重新编号社区，使社区序号确定且稳定。"""
    members: dict[int, list[str]] = {}
    for node in nodes:
        members.setdefault(community[node], []).append(node)
    # 以每个社区的最小成员排序，分配新序号
    ordered = sorted(members.values(), key=lambda ms: min(ms))
    remap: dict[str, int] = {}
    for new_id, ms in enumerate(ordered):
        for node in ms:
            remap[node] = new_id
    return remap


def _betweenness_centrality(
    adjacency: dict[str, set[str]], nodes: list[str]
) -> dict[str, float]:
    """无向无权图的中介度中心度（Brandes 算法，确定性）。

    中介度衡量节点作为「最短路径桥梁」的频次——资金网络中的**中介/通道账户**中介度高，
    是洗钱链路的关键节点。结果归一化到 [0,1]（除以 (n-1)(n-2)/2），便于规则按阈值引用。
    """
    n = len(nodes)
    betweenness: dict[str, float] = {v: 0.0 for v in nodes}
    if n < 3:
        return betweenness  # 少于 3 个点无中介可言

    for s in nodes:  # 确定性顺序
        # BFS 单源最短路（无权图）
        stack: list[str] = []
        predecessors: dict[str, list[str]] = {w: [] for w in nodes}
        sigma: dict[str, float] = {w: 0.0 for w in nodes}
        dist: dict[str, int] = {w: -1 for w in nodes}
        sigma[s] = 1.0
        dist[s] = 0
        queue = [s]
        qi = 0
        while qi < len(queue):
            v = queue[qi]
            qi += 1
            stack.append(v)
            for w in sorted(adjacency[v]):  # 确定性顺序
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    predecessors[w].append(v)
        # 回溯累积依赖
        delta: dict[str, float] = {w: 0.0 for w in nodes}
        while stack:
            w = stack.pop()
            for v in predecessors[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                betweenness[w] += delta[w]

    # 无向图每条最短路被计两次；归一化
    scale = 1.0 / ((n - 1) * (n - 2))  # 无向：2 / ((n-1)(n-2))，下方乘 2 抵消的一半
    return {v: round(betweenness[v] * scale, 6) for v in nodes}


def _node_in_cycle(adjacency: dict[str, set[str]], nodes: list[str]) -> dict[str, float]:
    """判定每个节点是否处于**环路**（资金闭环）中，返回 {节点: 1.0/0.0}。

    资金在一组对手间形成闭环（A→B→C→A）是典型的对敲/洗钱信号。无向图中，一个节点处于
    环上 ⟺ 它属于某个「非桥边构成的双连通分量」，等价于：删除所有桥后该节点仍有邻居，
    或所在连通分量满足 边数 ≥ 节点数（存在环）。本实现按连通分量判定 边数≥节点数 来标记
    分量内所有节点处于环结构（确定性、纯 Python）。
    """
    # 先求连通分量
    comp = _connected_components(adjacency, nodes)  # {node: 分量规模}
    # 重新按根聚合分量成员
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        root = x
        while parent.get(root, root) != root:
            root = parent[root]
        return root

    # 用并查集重建分量分组
    for node in nodes:
        parent.setdefault(node, node)
    for node in nodes:
        for nb in adjacency[node]:
            ra, rb = find(node), find(nb)
            if ra != rb:
                parent[ra if ra > rb else rb] = (rb if ra > rb else ra)

    members: dict[str, list[str]] = {}
    for node in nodes:
        members.setdefault(find(node), []).append(node)

    in_cycle: dict[str, float] = {v: 0.0 for v in nodes}
    for _root, ms in members.items():
        node_set = set(ms)
        # 分量内无向边数（每条边计一次）
        edge_count = 0
        for v in ms:
            edge_count += len(adjacency[v] & node_set)
        edge_count //= 2
        # 边数 ≥ 节点数 ⟹ 该连通分量存在环（树为 边数=节点数-1）
        if edge_count >= len(ms) and len(ms) >= 3:
            for v in ms:
                in_cycle[v] = 1.0
    return in_cycle


def build_graph_analytics_metrics(
    transactions: Iterable[CounterpartyTransaction],
    *,
    slice_ts: int,
    ref_name_prefix: str = DEFAULT_REF_NAME_PREFIX,
    ring_size_threshold: int = RING_SIZE_THRESHOLD_DEFAULT,
) -> list[CounterpartyMetric]:
    """从交易关系图提取团伙与中心度指标（确定性、幂等）。

    为每个节点产出七类指标：
    - ``{prefix}_community_size``：所属**连通分量**规模（粗粒度关联群组节点数）。
    - ``{prefix}_ring_flag``：连通分量规模 ≥ ``ring_size_threshold`` 时为 1.0，否则 0.0。
    - ``{prefix}_pagerank``：PageRank 中心度（保留 6 位小数）。
    - ``{prefix}_louvain_community``：Louvain **社区序号**（在大连通图内进一步切分的紧密团伙）。
    - ``{prefix}_louvain_size``：所属 Louvain 社区规模（细粒度团伙规模）。
    - ``{prefix}_betweenness``：中介度中心度（识别洗钱链路中的中介/通道账户，归一化 [0,1]）。
    - ``{prefix}_in_cycle``：是否处于资金闭环（环路/对敲洗钱信号）→ 1.0/0.0。

    结果按 (ref_name, dimension_key) 升序排序，保证与输入顺序无关（幂等）。
    """
    txns = list(transactions)
    adjacency, nodes = _collect_edges(txns)

    community_ref = validate_ref_name(f"{ref_name_prefix}_community_size")
    ring_ref = validate_ref_name(f"{ref_name_prefix}_ring_flag")
    pagerank_ref = validate_ref_name(f"{ref_name_prefix}_pagerank")
    louvain_comm_ref = validate_ref_name(f"{ref_name_prefix}_louvain_community")
    louvain_size_ref = validate_ref_name(f"{ref_name_prefix}_louvain_size")
    betweenness_ref = validate_ref_name(f"{ref_name_prefix}_betweenness")
    cycle_ref = validate_ref_name(f"{ref_name_prefix}_in_cycle")

    sizes = _connected_components(adjacency, nodes)
    ranks = _pagerank(adjacency, nodes)
    louvain = _louvain_communities(adjacency, nodes)
    louvain_sizes: dict[int, int] = {}
    for node in nodes:
        louvain_sizes[louvain[node]] = louvain_sizes.get(louvain[node], 0) + 1
    betweenness = _betweenness_centrality(adjacency, nodes)
    in_cycle = _node_in_cycle(adjacency, nodes)

    metrics: list[CounterpartyMetric] = []
    for node in nodes:
        size = float(sizes.get(node, 1))
        ring = 1.0 if size >= ring_size_threshold else 0.0
        rank = round(ranks.get(node, 0.0), 6)
        comm_id = float(louvain.get(node, 0))
        comm_size = float(louvain_sizes.get(louvain.get(node, 0), 1))
        metrics.append(CounterpartyMetric(community_ref, node, size, slice_ts))
        metrics.append(CounterpartyMetric(ring_ref, node, ring, slice_ts))
        metrics.append(CounterpartyMetric(pagerank_ref, node, rank, slice_ts))
        metrics.append(CounterpartyMetric(louvain_comm_ref, node, comm_id, slice_ts))
        metrics.append(CounterpartyMetric(louvain_size_ref, node, comm_size, slice_ts))
        metrics.append(
            CounterpartyMetric(betweenness_ref, node, betweenness.get(node, 0.0), slice_ts)
        )
        metrics.append(
            CounterpartyMetric(cycle_ref, node, in_cycle.get(node, 0.0), slice_ts)
        )

    metrics.sort(key=lambda m: (m.ref_name, m.dimension_key))
    return metrics
