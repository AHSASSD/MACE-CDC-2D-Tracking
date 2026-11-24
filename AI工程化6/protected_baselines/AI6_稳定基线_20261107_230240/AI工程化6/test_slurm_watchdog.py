"""Exercise outer recovery and stop policy without touching any Slurm job."""
import json
from pathlib import Path
import tempfile
from unittest import mock
import slurm_training_watchdog as guard

for authorize in [False,True]:
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);(root/'joint_training/joint_no_time_scratch_v1').mkdir(parents=True)
        (root/'joint_training/joint_no_time_scratch_v1/training_history.json').write_text('[{"epoch":29}]')
        (root/'slurm_training_launch.json').write_text('{"job_id":"123","srun_pid":11}')
        (root/'supervisor_policy.json').write_text(json.dumps({'auto_cancel_no_progress':authorize,'authorized_job_ids':['123']}))
        guard.ROOT=root
        with mock.patch('sys.argv',['guard','--job-id','123']),mock.patch.object(guard,'job_state',return_value=['user','RUNNING','node','2099-01-01T00:00:00']),mock.patch.object(guard,'alive',return_value=False),mock.patch.object(guard.time,'sleep'),mock.patch.object(guard.subprocess,'check_output',return_value='{"srun_pid":12}'),mock.patch.object(guard.subprocess,'check_call') as cancel:
            guard.main()
            assert cancel.call_count==int(authorize)
        record=json.loads((root/'joint_training/watchdog_123.json').read_text())
        assert record['status']==('allocation_released_after_repeated_failure' if authorize else 'needs_attention')
        assert len((root/'joint_training/watchdog_events.jsonl').read_text().splitlines())==4
assert guard.next_failure(29,30,2)==0
assert guard.next_failure(29,29,2)==3
from recovery_budget import update
with tempfile.TemporaryDirectory() as directory:
    root=Path(directory);(root/'joint_training').mkdir()
    assert update(root,'123',29)['no_progress_failures']==0
    for count in [1,2,3]:
        update(root,'123',29,begin=True)
        assert update(root,'123',29)['no_progress_failures']==count
        assert update(root,'123',29)['no_progress_failures']==count
    assert update(root,'123',29)['needs_attention']
    assert update(root,'123',30,begin=True)['no_progress_failures']==0
print('PASS: outer-step retry, progress resets counter, bounded no-progress retries, cancellation requires authorization')
