"""四条径迹混合分量的递归恢复检查，不依赖模型或 Excel。"""
import itertools
import numpy as np
import torch
from torch_geometric.data import Data
from STEP4_两级寻迹 import split_component
from STEP4_剩余混轨恢复 import recover_component, recover_remaining


def main():
    theta = np.linspace(-.8, .8, 5)
    x = np.concatenate([center + 60 * np.cos(theta) for center in [0, 150, 300, 450]])
    y = np.tile(100 + 60 * np.sin(theta), 4)
    edges, probabilities = [], []
    for group in range(4):
        for u, v in itertools.combinations(range(group * 5, group * 5 + 5), 2):
            edges.extend([(u, v), (v, u)])
            probabilities.extend([.95, .95])
    for u, v in [(4, 5), (9, 10), (14, 15)]:
        edges.extend([(u, v), (v, u)])
        probabilities.extend([.71, .71])
    data = Data(num_nodes=20, wire_x=torch.tensor(x, dtype=torch.float32),
                wire_y=torch.tensor(y, dtype=torch.float32), drift_d=torch.zeros(20),
                edge_index=torch.tensor(edges).t())
    probabilities = np.array(probabilities)
    baseline = split_component(data, list(range(20)), probabilities, .7, .2)
    recovered = recover_component(data, list(range(20)), probabilities, .7)
    assert not baseline, "The original two-edge cut should fail on four mixed circles"
    assert len(recovered) == 4 and all(t["rms"] <= .2 and t["hits"] == 5 for t in recovered)
    assert len({i for t in recovered for i in t["indices"]}) == 20
    frozen = recovered[:1]
    final = recover_remaining(data, frozen, probabilities)
    assert final[0] is frozen[0] and len(final) == 4
    assert recover_remaining(data, frozen, probabilities, min_probability=.96) == frozen
    print("Residual recursive checks passed")


if __name__ == "__main__":
    main()
