"""Verify the entire frozen file inventory after experiment work."""
import experiment_common as e
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import time


def main():
    started=time.time();original=Path(e.manifest['source']);bad=[];checked=0
    def check(item):
        rel,row=item
        return rel,e.sha(e.BASELINE/rel)==row['sha256'],e.sha(original/rel)==row['sha256']
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(check,item) for item in e.manifest['sha256'].items()]):
            rel,copy_ok,original_ok=future.result();checked+=1
            if not copy_ok or not original_ok:bad.append(dict(path=rel,backup_ok=copy_ok,original_ok=original_ok))
    result=dict(files_checked=checked,backup_all_bytes_unchanged=not any(not r['backup_ok'] for r in bad),
                original_stable_files_all_unchanged=not any(not r['original_ok'] for r in bad),differences=bad,
                new_experiment_files_not_in_frozen_inventory=True,seconds=time.time()-started,baseline=e.PIN)
    e.save_json(e.EXP/'stable_integrity_verification.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
    if bad:raise RuntimeError('Frozen file differences found; see verification, do not overwrite changes')


if __name__=='__main__':main()
