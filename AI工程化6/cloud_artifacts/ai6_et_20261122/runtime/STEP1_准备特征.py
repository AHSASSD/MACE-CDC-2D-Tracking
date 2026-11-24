import argparse,json,time
from pathlib import Path
import torch
from et_data import prepare


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--weight',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--splits',type=Path,required=True)
    p.add_argument('--mode',choices=['smoke','pilot','full','validation'],default='pilot');p.add_argument('--device',default='cuda')
    args=p.parse_args();torch.set_num_threads(1);s=json.loads(args.splits.read_text(encoding='utf-8'))
    if args.mode=='smoke':ids=s['pilot_train_event_ids'][:7]+s['stress_train_event_ids'][:1]
    elif args.mode=='pilot':ids=s['pilot_train_event_ids']+s['periodic_member_validation_event_ids']
    elif args.mode=='full':ids=s['original_train_event_ids']+s['periodic_member_validation_event_ids']
    else:ids=s['original_validation_event_ids']
    prepare(args.data,args.weight,args.cache,list(dict.fromkeys(ids)),device=args.device)


if __name__=='__main__':main()
