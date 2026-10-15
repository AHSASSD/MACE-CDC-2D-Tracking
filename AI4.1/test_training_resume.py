"""A saved AdamW/scheduler/dropout state reproduces the next training step."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch

from common import ROOT


def main():
    trainer = importlib.import_module('STEP2_训练模型')
    torch.set_num_threads(1)
    torch.manual_seed(123)

    def setup():
        model = torch.nn.Sequential(torch.nn.Linear(6, 12), torch.nn.Dropout(.25), torch.nn.Linear(12, 1))
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=.5)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        return model, optimizer, scheduler, scaler

    model, optimizer, scheduler, scaler = setup()
    x, y = torch.randn(8, 6), torch.randn(8, 1)

    def step(model, optimizer, scheduler, value):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        scheduler.step(value)

    step(model, optimizer, scheduler, .10)
    step(model, optimizer, scheduler, .11)
    with tempfile.TemporaryDirectory(dir=ROOT / 'whole_event_retraining') as temporary:
        path = Path(temporary) / 'state.pth'
        trainer.save_atomic({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                             'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                             'torch_rng': torch.get_rng_state(), 'cuda_rng': []}, path)
        step(model, optimizer, scheduler, .12)
        expected = {k: v.clone() for k, v in model.state_dict().items()}
        expected_random = torch.rand(5)
        restored_model, restored_optimizer, restored_scheduler, restored_scaler = setup()
        state = torch.load(path, map_location='cpu', weights_only=True)
        trainer.restore_training_state(state, restored_model, restored_optimizer, restored_scheduler, restored_scaler)
        step(restored_model, restored_optimizer, restored_scheduler, .12)
        assert all(torch.equal(expected[k], v) for k, v in restored_model.state_dict().items())
        assert restored_optimizer.param_groups[0]['lr'] == optimizer.param_groups[0]['lr'] == 1.5e-4
        assert restored_scheduler.state_dict() == scheduler.state_dict()
        assert torch.equal(torch.rand(5), expected_random)
        assert not path.with_suffix('.pth.tmp').exists()
    print('Resume self-check passed: weights, AdamW, scheduler, dropout RNG; no production weights changed')
    directory = ROOT / 'whole_event_retraining'
    train_graphs = torch.load(next((directory / 'dataset_train.pt.parts').glob('part_*.pt')), weights_only=False)[:2]
    val_graphs = torch.load(next((directory / 'dataset_validation.pt.parts').glob('part_*.pt')), weights_only=False)[:2]
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        target = Path(temporary)
        assert target.resolve().parent == directory.resolve()
        torch.save(train_graphs, target / 'dataset_train.pt')
        torch.save(val_graphs, target / 'dataset_validation.pt')
        (target / 'dataset_metadata.json').write_bytes((directory / 'dataset_metadata.json').read_bytes())
        split = {'train_evt_ids': [int(g.event_id) for g in train_graphs],
                 'val_evt_ids': [int(g.event_id) for g in val_graphs], 'test_evt_ids': []}
        (target / 'event_split.json').write_text(json.dumps(split), encoding='utf-8')
        base = [sys.executable, str(ROOT / 'STEP2_训练模型.py'), '--features', 'no-time',
                '--dataset-root', str(target), '--run-tag', 'resume_test']
        env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
               'MACE_BATCH_SIZE': '2', 'MACE_VAL_BATCH_SIZE': '2', 'MACE_ACCUMULATION_STEPS': '1'}
        subprocess.run(base + ['--epochs', '1', '--lr', '0.0003', '--init-model',
                               str(directory / 'best_interaction_network_no_time_finetune_whole_events_sigma05.pth')],
                       check=True, env=env)
        checkpoint = target / 'training_state_no_time_resume_test.pth'
        subprocess.run(base + ['--epochs', '2', '--resume', str(checkpoint)], check=True, env=env)
        state = torch.load(checkpoint, weights_only=True, map_location='cpu')
        assert state['epoch'] == 2 and state['initial_learning_rate'] == .0003
        assert state['optimizer']['state'] and state['best_model']
        metadata = json.loads((target / 'model_metadata_no_time_resume_test.json').read_text(encoding='utf-8'))
        assert metadata['optimizer_state_reused'] is True
    print('Training CLI check passed: 2+2 toy graphs, one epoch then resumed next epoch')


if __name__ == '__main__':
    main()
