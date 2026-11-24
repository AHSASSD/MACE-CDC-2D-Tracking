"""Read-only preparation gate, runs on login/compute Python3.6+ without torch."""
import argparse,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--mode',choices=['pilot','full'],required=True);args=p.parse_args()
    root=args.root;s=json.loads((root/'runtime/split_manifest.json').read_text())
    ids=s['pilot_train_event_ids'] if args.mode=='pilot' else s['original_train_event_ids']
    all_ids=ids+s['periodic_member_validation_event_ids'];rows=[json.loads((root/'cache/stats'/('event_%05d.json'%i)).read_text()) for i in all_ids]
    reference=sum(x['triplet_pruning']['reference_reconstructible'] for x in rows);lost=sum(x['triplet_pruning']['lost_by_bounded_triplets'] for x in rows)
    result=dict(mode=args.mode,events=len(rows),reference_tracks=reference,lost_by_triplet_pruning=lost,
        pruning_loss_fraction=lost/max(reference,1),max_triplets=max(x['triplets'] for x in rows),passed=lost/max(reference,1)<=.002)
    (root/('cache_gate_'+args.mode+'.json')).write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    if not result['passed']:raise RuntimeError('Bounded triplet policy exceeds planned0.2 percentage-point oracle coverage loss')


if __name__=='__main__':main()
