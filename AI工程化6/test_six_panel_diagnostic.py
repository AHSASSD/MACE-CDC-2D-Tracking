"""Validate diagnostic identities, archived reconstruction and the Tk case selector."""
import importlib,json
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from PIL import Image
import numpy as np
m=importlib.import_module('STEP19_六联图可视化诊断')
cases=m.catalog();assert len(cases)==223 and len({(c['event'],c['truth']) for c in cases})==223
for model in ['baseline','balanced_012']:
    packet=m.read(m.OUT/f'event_06978_T1456_{model}.json')
    assert Image.open(m.OUT/f'event_06978_T1456_{model}.png').size==(2660,1680)
    original=m.read(m.EXP/'attribution_v1/events/event_06978.json') if model=='baseline' else m.read(m.review.OUT/'case_attribution_20261109/events'/model/'event_06978.json')
    found={g['truth'] for g in packet['groups'] if g['status']=='matched'}
    if model=='baseline':assert found==set(original['baseline_matched'])
    else:
        assert len(found)==original['counts']['matched_tracks']
        assert sum(g['status']=='fake' for g in packet['groups'])==original['counts'].get('fake_candidates',0)
    circle=packet['geometry_circle']
    for p in packet['intersections']:
        assert abs(np.hypot(p['x']-circle[0],p['y']-circle[1])-circle[2]-p['radius_offset_mm'])<1e-4
def loop(self,*a,**k):
    self.update_idletasks()
    def descendants(w):
        return [w]+[v for child in w.winfo_children() for v in descendants(child)]
    widgets=descendants(self);tree=next(w for w in widgets if isinstance(w,ttk.Treeview))
    assert len(tree.get_children())==223
    assert any(isinstance(w,ttk.Combobox) and len(w['values'])==4 for w in widgets)
    self.destroy()
tk.Tk.mainloop=loop
m.gui()
result=dict(cases=223,layout='2 rows x 3 columns',image_size=[2660,1680],archived_reconstruction_matches=True,cylinder_intersections_verified=True,gui_case_selector='passed')
m.e.save_json(m.OUT/'verification.json',result);print(json.dumps(result))
