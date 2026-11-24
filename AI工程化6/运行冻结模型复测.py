"""Continue after STEP5 validation completes; stop on any subprocess failure."""
import argparse
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',required=True);args=parser.parse_args()
    root=Path(__file__).resolve().parent;snapshot=root/'joint_training'/args.snapshot
    deadline=time.time()+900
    while not (snapshot/'two_stage_validation_v1/summary.json').exists():
        if time.time()>deadline:raise TimeoutError('STEP5 validation has not completed')
        time.sleep(5)
    for script,extra in [('STEP8_寻迹改进评估.py',['--workers','2']),('STEP9_固定方案测试.py',['--workers','4'])]:
        subprocess.run([sys.executable,'-u',str(root/script),'--snapshot',args.snapshot]+extra,cwd=root,check=True)


if __name__=='__main__':main()
