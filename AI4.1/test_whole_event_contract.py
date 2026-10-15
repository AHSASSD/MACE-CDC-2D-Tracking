"""Check legacy/new metadata and a real new-data training batch."""
import importlib
import json
from pathlib import Path
import tempfile

import torch
from torch_geometric.loader import DataLoader

from common import ROOT, DATASET_META_FILE, InteractionNetwork, model_dimensions, select_features
from STEP1_完整事件数据集 import self_check


def main():
    trainer = importlib.import_module('STEP2_训练模型')
    directory = ROOT / 'whole_event_retraining'
    for variant in ['no-time', 'time']:
        trainer.validate_dataset_metadata(DATASET_META_FILE, variant)
        trainer.validate_dataset_metadata(directory / 'dataset_metadata.json', variant)
        model = InteractionNetwork(*model_dimensions(variant))
        path = ROOT / f"best_interaction_network_{variant.replace('-', '_')}_clean_physics_v2.pth"
        info = trainer.load_initial_weights(model, path)
        saved = torch.load(path, weights_only=True, map_location='cpu')
        assert all(torch.equal(model.state_dict()[key], value) for key, value in saved.items())
        assert all(p.requires_grad for p in model.parameters()) and len(info['sha256']) == 64
    try:
        trainer.load_initial_weights(InteractionNetwork(*model_dimensions('time')),
            ROOT / 'best_interaction_network_no_time_clean_physics_v2.pth')
    except RuntimeError:
        pass
    else:
        raise AssertionError('Mismatched feature-model weights accepted')
    metadata = json.loads((directory / 'dataset_metadata.json').read_text(encoding='utf-8'))
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        checkpoint = Path(temporary) / 'weights.pth'
        trainer.save_weights(model, checkpoint)
        restored = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert all(torch.equal(model.state_dict()[key], value) for key, value in restored.items())
        assert not checkpoint.with_suffix('.pth.tmp').exists()
        path = Path(temporary) / 'invalid.json'
        metadata['timing']['basis'] = 'per_particle_truth_t0'
        path.write_text(json.dumps(metadata), encoding='utf-8')
        try:
            trainer.validate_dataset_metadata(path, 'time')
        except ValueError:
            pass
        else:
            raise AssertionError('Truth-based time origin accepted')
    self_check()
    torch.set_num_threads(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graphs = torch.load(next((directory / 'dataset_train.pt.parts').glob('part_*.pt')), weights_only=False)
    batch = next(iter(DataLoader(graphs[:2], batch_size=2,
                exclude_keys=['truth_ptr', 'truth_pairs', 'all_truth_pairs']))).to(device)
    for variant in ['no-time', 'time']:
        model = InteractionNetwork(*model_dimensions(variant)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        x, edge = select_features(batch, variant)
        loss = trainer.FocalLoss()(model(x, batch.edge_index, edge), batch.y)
        loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
        print('Training-batch check passed:', variant, device, float(loss.detach()), flush=True)
        del model, optimizer, loss
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    print('Legacy and whole-event contracts passed; no weights saved', flush=True)


if __name__ == '__main__':
    main()
