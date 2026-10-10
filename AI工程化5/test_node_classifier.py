import torch

from STEP2_训练节点分类器 import NodeStateClassifier, binary_uncertainty


def main():
    model = NodeStateClassifier(node_in=13, edge_in=8)
    x = torch.zeros(5, 7)
    node_context = torch.zeros(5, 6)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    edge_attr = torch.zeros(4, 6)
    edge_context = torch.zeros(4, 2)
    noise, shared = model(x, edge_index, edge_attr, node_context, edge_context)
    assert noise.shape == shared.shape == (5,)
    uncertainty = torch.maximum(binary_uncertainty(noise), binary_uncertainty(shared))
    assert torch.isfinite(uncertainty).all()
    assert bool(((0.0 <= uncertainty) & (uncertainty <= 1.0)).all())
    print("node classifier self-check passed")


if __name__ == "__main__":
    main()
