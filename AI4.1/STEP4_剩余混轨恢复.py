"""只恢复首轮未使用的 hit；递归拆分失败分量，已接受径迹冻结。"""
import networkx as nx

from STEP4_两级寻迹 import components, score, trim_outliers


def recover_component(data, nodes, probabilities, threshold, depth=0, budget=None):
    # shortcut: 每分量最多40次候选检查和4层拆分，扩大预算前需重新验证耗时。
    budget = [40] if budget is None else budget
    if budget[0] <= 0:
        return []
    budget[0] -= 1
    candidate = trim_outliers(data, nodes, probabilities, threshold, .20, "residual_recursive")
    if candidate is not None:
        return [candidate] if candidate["mean_probability"] >= .80 else []
    if depth >= 4 or len(nodes) < 10:
        return []
    members = set(nodes)
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    for (u, v), probability in zip(data.edge_index.t().tolist(), probabilities):
        if u in members and v in members and probability >= threshold:
            graph.add_edge(u, v, weight=float(probability))
    tree = nx.maximum_spanning_tree(graph)
    best = []
    for u, v, attrs in sorted(tree.edges(data=True), key=lambda e: e[2]["weight"])[:3]:
        tree.remove_edge(u, v)
        parts = [sorted(p) for p in nx.connected_components(tree) if len(p) >= 5]
        tree.add_edge(u, v, **attrs)
        if len(parts) < 2:
            continue
        candidates = [track for part in parts for track in
                      recover_component(data, part, probabilities, threshold, depth + 1, budget)]
        if sum(score(t) for t in candidates) > sum(score(t) for t in best):
            best = candidates
    return best


def recover_remaining(data, tracks, probabilities, threshold=.70, min_probability=.80):
    used = {i for track in tracks for i in track["indices"]}
    final = list(tracks)
    for component in components(data, probabilities, threshold, set(range(data.num_nodes)) - used):
        for track in recover_component(data, component, probabilities, threshold):
            if track["mean_probability"] >= min_probability and not used.intersection(track["indices"]):
                track["score"] = score(track)
                final.append(track)
                used.update(track["indices"])
    return final
