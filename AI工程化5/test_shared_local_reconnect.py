import numpy as np
import torch
from torch_geometric.data import Data

from STEP4_shared局部重连 import anchor_fragments


def main():
    data = Data(edge_index=torch.tensor([[0, 1, 3], [1, 2, 4]]), num_nodes=5)
    fragments = anchor_fragments(data, np.array([0.9, 0.9, 0.9]), set(range(5)), 0.55)
    assert sorted(map(len, fragments)) == [2, 3]
    print("shared local reconnect self-check passed")


if __name__ == "__main__":
    main()
