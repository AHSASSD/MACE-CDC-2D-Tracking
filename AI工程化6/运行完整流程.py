"""Local persistent pipeline: full scratch training, then frozen evaluation."""
import subprocess
import sys
import time
from common import *
from training_health import run_monitored

def supervised_training(cmd,log,root,max_restarts=2,runner=run_monitored,delay=10):
    import os
    from datetime import datetime
    for attempt in range(max_restarts+1):
        try:return runner(cmd,log,root)
        except subprocess.CalledProcessError as exc:
            if exc.returncode!=-9:raise
            checkpoint=root/'joint_training/joint_no_time_scratch_v1/training_state_joint_no_time.pth'
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            job=os.getenv('SLURM_JOB_ID')
            if job:
                from recovery_budget import update
                budget=update(root,job,state['epoch'])
                if budget['needs_attention']:raise
            if attempt==max_restarts:raise
            end=int(os.getenv('SLURM_JOB_END_TIME','0'))
            job_record=root/'current_job.json'
            if not end and job_record.exists():
                current=json.loads(job_record.read_text())
                if str(current.get('job_id'))==os.getenv('SLURM_JOB_ID'):
                    end=int(datetime.fromisoformat(current['end_time']).timestamp())
            if end and end-time.time()<90:raise
            if job:update(root,job,state['epoch'],begin=True)
            record={'unix':time.time(),'signal':9,'attempt':attempt+1,'max_restarts':max_restarts,'checkpoint_epoch':state['epoch'],'next_epoch':state['epoch']+1,'slurm_job':os.getenv('SLURM_JOB_ID'),'slurm_step':os.getenv('SLURM_STEP_ID')}
            with (root/'joint_training/recovery_events.jsonl').open('a') as events:events.write(json.dumps(record)+'\n')
            log.write('RECOVER_SIGKILL '+json.dumps(record)+'\n');log.flush()
            save_json(root/'joint_training/pipeline_status.json',{'status':'recovering','details':record})
            time.sleep(delay)
            save_json(root/'joint_training/pipeline_status.json',{'status':'training','last_recovery':record})

def main():
    tag='joint_no_time_scratch_v1';MODEL_ROOT.mkdir(exist_ok=True)
    checkpoint=MODEL_ROOT/tag/'training_state_joint_no_time.pth'
    cmd=[sys.executable,'-u',str(ROOT/'STEP2_训练模型.py'),'--run-tag',tag,'--epochs','200']
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        cmd+=['--resume',str(checkpoint),'--chunk-size',str(state['config']['chunk_size'])]
    save_json(MODEL_ROOT/'pipeline_status.json',{'status':'training','run_tag':tag,'target_epochs':200,'started_unix':time.time()})
    try:
        with (MODEL_ROOT/f'train_{tag}.log').open('a',encoding='utf-8') as log:
            supervised_training(cmd,log,ROOT)
            save_json(MODEL_ROOT/'pipeline_status.json',{'status':'evaluating','run_tag':tag,'updated_unix':time.time()})
            for split in ['validation','test']:
                if not (MODEL_ROOT/tag/f'edge_evaluation_{split}.json').exists():
                    subprocess.run([sys.executable,'-u',str(ROOT/'STEP3_推理评估.py'),'--run-tag',tag,'--split',split],stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
        save_json(MODEL_ROOT/'pipeline_status.json',{'status':'complete','run_tag':tag,'completed_unix':time.time()})
    except BaseException as exc:
        save_json(MODEL_ROOT/'pipeline_status.json',{'status':'failed','error':str(exc),'updated_unix':time.time()});raise

if __name__=='__main__':main()
