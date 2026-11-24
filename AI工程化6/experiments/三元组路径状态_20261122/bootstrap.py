from pathlib import Path
import os,sys
HERE=Path(__file__).resolve().parent
BASE=Path(os.environ.get('AI6_T_BASE',str(HERE.parent/'双路线寻迹_20261122')))
sys.path.insert(0,str(BASE))
