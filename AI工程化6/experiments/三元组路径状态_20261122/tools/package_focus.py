from pathlib import Path
import ast,zipfile,json,hashlib
ROOT=Path(__file__).resolve().parent.parent
FILES=['bootstrap.py','path_decode.py','focus_data.py','train.py','evaluate.py','validate_members.py','supervisor.py','check_states.py']

def main():
    hashes={}
    with zipfile.ZipFile(ROOT/'cloud_runtime.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in FILES:
            p=ROOT/name;ast.parse(p.read_text(encoding='utf-8'),feature_version=(3,10));hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest();z.write(p,'runtime/'+name)
        for name in ['step.sh','train_focus.sbatch']:z.write(ROOT/name,name)
    (ROOT/'packaged_sources.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8');print('PACKAGED',len(FILES),'modules; runtime source only')

if __name__=='__main__':main()
