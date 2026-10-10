import torch
from torch_geometric.data import Data

from STEP1_生成AI5契约数据集 import apply_contract_labels


def main():
    graph = Data(
        edge_index=torch.tensor([[0, 0], [1, 2]], dtype=torch.long),
        truth_ptr=torch.tensor([0, 2, 3, 4], dtype=torch.long),
        truth_pairs=torch.tensor([[1, 1], [1, 2], [1, 2], [1, 1]], dtype=torch.long),
        num_nodes=3,
    )
    apply_contract_labels(graph)
    assert graph.y.tolist() == [0.0, 1.0]
    assert graph.edge_shared_compatible.tolist() == [True, True]
    assert graph.edge_shared_ambiguous.tolist() == [True, False]
    print("AI5 contract self-check passed")


if __name__ == "__main__":
    main()
