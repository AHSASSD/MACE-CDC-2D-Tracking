"""Recovered AI4 six-panel diagnostic, adapted to pinned joint axial/stereo AI6 results."""
from pathlib import Path
import argparse, csv, json, sys, hashlib, html, subprocess
from collections import Counter
ROOT=Path(__file__).resolve().parent
EXP=ROOT/'experiments/复杂事件归因_20261107'
sys.path.insert(0,str(EXP));sys.dont_write_bytecode=True
import experiment_common as e
sys.path.insert(0,str(EXP/'parallel_20261109'))
import STEP7_训练结果与同流程复测 as review
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from two_stage_core import fit_2d_circle
from coarse_3d_membership import finite_cylinder_intersections

OUT=ROOT/'visual_diagnosis_20261121'
MODELS={'baseline':'原稳定 epoch126','control_012':'原Focal微调12轮','balanced_012':'整轨辅助微调12轮','balanced_015':'整轨辅助微调15轮'}
LABELS={'broadly_low_scores':'真边普遍低分','pure_connected_at_existing_relaxed_cut':'纯真边已连通但漏检','mixed_fragmentation':'混合碎裂','few_weak_connectors':'少数关键连接低分','candidate_graph_insufficient':'AA候选图不足'}


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def catalog():
    cases=[]
    for eid in e.IDS:
        for t in read(EXP/'edge_score_diagnosis_v1/events'/f'event_{eid:05d}.json')['tracks']:
            if not t['found']:cases.append(dict(event=eid,truth=t['truth_id'],category=t['category'],hits=t['axial_hits']))
    return cases


def load(eid,model):
    e.verify_pins()
    with np.load(e.DATA_ROOT/'events'/f'event_{eid:05d}.npz') as f:
        cells,drift,truth,hit=f['cell_id'],f['d'],f['truth'],f['hit_id']
    geo=np.load(e.DATA_ROOT/'wire_geometry.npy')[cells];stereo=np.linalg.norm(geo[:,2:4],axis=1)>1e-9
    cache=e.SNAPSHOT/('validation_prediction_cache_'+e.CONTRACT['weight_sha256'][:12]) if model=='baseline' else review.OUT/'prediction_cache'/model
    with np.load(cache/f'event_{eid:05d}.npz') as f:pairs,p=f['pairs'],f['probabilities']
    if model=='baseline':
        tracks=read(e.SNAPSHOT/'coarse_3d_inputs_validation'/f'event_{eid:05d}.json')['tracks']
        coarse=read(e.VALIDATION/'events'/f'event_{eid:05d}.json')['candidates']['guarded_loose']
        aa=[[i for i in t['indices'] if not stereo[i]] for t in tracks]
        confirmed=[t['confirmed_stereo_indices'] for t in coarse]
        fits=[t['fit'] for t in tracks]
        weight=e.CONTRACT['weight_sha256']
    else:
        file=OUT/'cache'/model/f'event_{eid:05d}.json';file.parent.mkdir(parents=True,exist_ok=True)
        if file.exists():packet=read(file)
        else:
            review.initialize();packet=review.event_evaluate(model,eid)
            packet['diagnostic_fits']=[]
            for group in packet['axial_members']:
                xy=geo[group,:2];fit=fit_2d_circle(xy[:,0],xy[:,1],drift[group]);packet['diagnostic_fits'].append(fit)
            e.save_json(file,packet)
        aa=packet['axial_members'];confirmed=packet['confirmed_stereo_members'];fits=packet['diagnostic_fits']
        arm,epoch=model.rsplit('_',1);weight=e.sha(review.OUT/'cloud'/arm/f'epoch_{int(epoch):03d}.pth')
    groups=[];seen=set();sizes=Counter(truth[~stereo])
    for number,(indices,fit,ss) in enumerate(zip(aa,fits,confirmed),1):
        tid,n=Counter(truth[indices]).most_common(1)[0]
        purity=n/len(indices);completeness=n/sizes[tid]
        valid=tid>=0 and sizes[tid]>=5 and purity>=.8 and completeness>=.5
        status='fake' if not valid else 'clone' if tid in seen else 'matched'
        if valid:seen.add(tid)
        rms=float(np.sqrt(np.mean((np.abs(np.linalg.norm(geo[indices,:2]-np.array(fit[:2]),axis=1)-fit[2])-drift[indices])**2))) if fit else None
        groups.append(dict(number=number,indices=indices,fit=fit,confirmed=ss,seed_ready=bool(len(np.unique(cells[ss]))>=2),truth=int(tid),purity=purity,completeness=completeness,status=status,rms=rms))
    return cells,drift,truth,hit,geo,stereo,pairs,p,groups,weight


def arc(fit,xy):
    center=np.array(fit[:2]);angles=np.sort(np.mod(np.arctan2(xy[:,1]-center[1],xy[:,0]-center[0]),2*np.pi))
    gaps=np.diff(np.r_[angles,angles[0]+2*np.pi]);j=int(gaps.argmax());a=angles[(j+1)%len(angles)]
    phase=np.linspace(a-.04,a+2*np.pi-gaps[j]+.04,240)
    return center+fit[2]*np.column_stack([np.cos(phase),np.sin(phase)])


def draw(eid,tid,model,strict=.7,relaxed=.575):
    OUT.mkdir(exist_ok=True);cells,d,truth,hit,geo,stereo,pairs,p,groups,weight=load(eid,model)
    aa=np.flatnonzero(~stereo);target=np.flatnonzero((truth==tid)&~stereo)
    if len(target)<5:raise ValueError('目标真径迹少于5个直丝hit，不在当前评价口径内')
    ids=sorted(set(truth[aa]));colors={t:plt.get_cmap('turbo')((i+.5)/len(ids)) for i,t in enumerate(ids)}
    matched={g['truth'] for g in groups if g['status']=='matched'}
    sizes=Counter(truth[aa]);eligible={int(t) for t,n in sizes.items() if t>=0 and n>=5}
    missed=eligible-matched
    seed_eligible={t for t in eligible if len(np.unique(cells[(truth==t)&stereo]))>=2}
    seed_found=set()
    for g in groups:
        ss=np.asarray(g['confirmed'],dtype=int)
        good=ss[truth[ss]==g['truth']]
        if g.get('seed_ready',False) and g['status'] in ('matched','clone') and g['truth'] in seed_eligible and len(np.unique(cells[good]))>=2 and len(good)/max(len(ss),1)>=.8:seed_found.add(g['truth'])
    missing_3d_only=(seed_eligible-seed_found)&matched
    reference_fit=fit_2d_circle(geo[target,0],geo[target,1],d[target])
    reference_rms=float(np.sqrt(np.mean((np.abs(np.linalg.norm(geo[target,:2]-np.array(reference_fit[:2]),axis=1)-reference_fit[2])-d[target])**2))) if reference_fit else None
    true=(truth[pairs[0]]==truth[pairs[1]])&(truth[pairs[0]]>=0)
    axial_edges=~stereo[pairs[0]]&~stereo[pairs[1]]
    masks=[(true&(p>=strict),'#159947','-', '真边 ≥ 严格阈值'),(true&(p>=relaxed)&(p<strict),'#d79a00','--','真边：宽松～严格'),(~true&(p>=relaxed),'#dd3030','-','跨真轨假边'),(true&(p<relaxed),'#777777',':','真边低于宽松阈值')]
    fig,axes=plt.subplots(2,3,figsize=(19,12));xy=geo[:,:2]
    low=xy[target].min(0);high=xy[target].max(0);center=(low+high)/2;half=max(float((high-low).max())/2+25,40)
    extent=float(np.abs(xy[aa]).max())+30
    for row in range(2):
        for ax in axes[row]:
            ax.scatter(*xy[aa].T,s=6,c='#c7c7c7',alpha=.4);ax.set_aspect('equal');ax.grid(alpha=.15);ax.set(xlabel='x / mm',ylabel='y / mm')
            ax.set(xlim=(-extent,extent),ylim=(-extent,extent)) if row==0 else ax.set(xlim=(center[0]-half,center[0]+half),ylim=(center[1]-half,center[1]+half))
        for t in ids if row==0 else [tid]:
            indices=np.flatnonzero((truth==t)&~stereo);axes[row,0].scatter(*xy[indices].T,s=19,c=[colors[t]],marker='x')
            axes[row,0].annotate(f'T{t}',xy[indices[0]],fontsize=6,color=colors[t])
            if row:
                for i in indices:axes[row,0].add_patch(Circle(xy[i],d[i],fill=False,color=colors[t],lw=.7))
                for cell in np.unique(cells[indices]):
                    ix=indices[cells[indices]==cell];i=ix[0]
                    axes[row,0].annotate(f'Cell{cell} ×{len(ix)}',xy[i],xytext=(3,3),textcoords='offset points',fontsize=6)
                axes[row,0].text(.02,.02,f'{len(indices)}个独立hit / {len(np.unique(cells[indices]))}根丝\n真值分组诊断圆RMS={reference_rms:.2f}mm（仅离线参照）' if reference_rms is not None else '无有效诊断圆',transform=axes[row,0].transAxes,fontsize=8,bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
        visible=axial_edges if row==0 else axial_edges&((truth[pairs[0]]==tid)|(truth[pairs[1]]==tid))
        for mask,color,style,label in masks:
            axes[row,1].add_collection(LineCollection(xy[pairs[:,mask&visible].T],colors=color,linestyles=style,linewidths=.65 if row else .3,alpha=.75 if row else .25,label=label))
        axes[row,1].scatter(*xy[target if row else aa].T,s=10,c='#222222');axes[row,1].legend(fontsize=7,loc='upper right')
        used=set();handles=[]
        for g in groups:
            indices=g['indices'];used.update(indices)
            if row and not set(indices)&set(target):continue
            status=g['status'];color=colors[g['truth']] if status=='matched' else '#d62d2d' if status=='fake' else '#9c30b6'
            style={'matched':'-','fake':'--','clone':':'}[status]
            if g['fit']:
                curve=arc(g['fit'],xy[indices]);axes[row,2].plot(*curve.T,color=color,ls=style,lw=1)
            axes[row,2].scatter(*xy[indices].T,s=18,c=[color],marker={'matched':'o','fake':'s','clone':'D'}[status])
            axes[row,2].annotate(f'C{g["number"]}/T{g["truth"]}',xy[indices[0]],fontsize=6,color=color)
            if row:
                for i in indices:axes[row,2].add_patch(Circle(xy[i],d[i],fill=False,color=color,lw=.5,alpha=.6))
                handles.append(Line2D([],[],color=color,ls=style,label=f'C{g["number"]} {status} 纯度{g["purity"]:.2f} 完整度{g["completeness"]:.2f}\n诊断RMS={g["rms"]:.2f}mm' if g['rms'] is not None else '无有效拟合'))
        unused=sorted(set(target if row else aa)-used)
        if unused:axes[row,2].scatter(*xy[unused].T,s=28,c='black',marker='x')
        if not row or handles:axes[row,2].legend(handles=handles if row else [Line2D([],[],color='green',label='匹配（与真轨同色）'),Line2D([],[],color='#d62d2d',ls='--',label='假轨'),Line2D([],[],color='#9c30b6',ls=':',label='克隆'),Line2D([],[],color='black',marker='x',ls='',label='未分配hit')],fontsize=7,loc='upper right')
        for col,ax in enumerate(axes[row]):
            for label_set,color,marker,prefix in [(missed,'#c2185b','o','漏轨'),(missing_3d_only,'#b86a00','s','三维未成种子')]:
                visible_t=sorted(label_set) if row==0 else ([tid] if tid in label_set else [])
                for number,t in enumerate(visible_t):
                    ix=np.flatnonzero((truth==t)&~stereo)
                    ax.scatter(*xy[ix].T,s=65,facecolors='none',edgecolors=color,marker=marker,lw=1.4,zorder=8)
                    anchor=xy[ix].mean(0)
                    ax.annotate(f'{prefix} T{t}',anchor,xytext=(12,15+14*(number%3)),textcoords='offset points',color=color,fontsize=8,fontweight='bold',zorder=10,
                                bbox=dict(boxstyle='round,pad=.2',fc='white',ec=color,alpha=.9),arrowprops=dict(arrowstyle='->',color=color,lw=.8))
            if row:
                label=f'漏轨 T{tid}' if tid in missed else f'T{tid}：直丝已找到，三维未成种子' if tid in missing_3d_only else f'T{tid}：已找到'
                ax.text(.02,.98,label,transform=ax.transAxes,va='top',fontsize=9,fontweight='bold',color='#c2185b' if tid in missed else '#b86a00' if tid in missing_3d_only else '#16834a',bbox=dict(fc='white',alpha=.9,ec='none'),zorder=11)
    for i,title in enumerate(['真值分组：直丝丝中心 ×（不是实际hit坐标）','模型AA连边：双向概率平均值','重建分组 + 诊断拟合圆弧']):
        axes[0,i].set_title(title,fontsize=10);axes[1,i].set_title(f'局部放大：TrkID={tid}',fontsize=10)
    matched={g['truth'] for g in groups if g['status']=='matched'}
    record=next((c for c in catalog() if c['event']==eid and c['truth']==tid),None)
    cause=LABELS[record['category']] if record else '自选径迹'
    fig.suptitle(f'EvtID={eid} | {MODELS[model]} | T{tid}：{cause} | 当前{"已找回" if tid in matched else "仍未匹配"} | 匹配{len(matched)} 假轨{sum(g["status"]=="fake" for g in groups)} 克隆{sum(g["status"]=="clone" for g in groups)} | 显示阈值{strict:g}/{relaxed:g}',fontsize=12)
    fig.text(.02,.055,'直丝段漏轨（紫红圈/箭头）：'+(', '.join(f'T{t}' for t in sorted(missed)) or '无'),fontsize=10,color='#c2185b',weight='bold',wrap=True)
    fig.text(.02,.038,'直丝已找到但粗三维未成种子（橙方框）：'+(', '.join(f'T{t}' for t in sorted(missing_3d_only)) or '无'),fontsize=9,color='#b86a00',wrap=True)
    fig.text(.02,.009,'上排全事件，下排局部；黑叉=未分配hit，不等于漏轨。真值仅用于离线诊断；丝中心不是实际hit坐标。\n漏轨按至少5个直丝hit、纯度≥80%且完整度≥50%判断；橙框另按确认斜丝/种子条件判断。阈值仅改连边显示。',fontsize=9)
    fig.tight_layout(rect=(0,.09,1,.95))
    stem=f'event_{eid:05d}_T{tid}_{model}';path=OUT/(stem+'.png');fig.savefig(path,dpi=140);plt.close(fig)
    focus=[g for g in groups if set(g['indices'])&set(target)]
    best=max(focus,key=lambda g:len(set(g['indices'])&set(target))) if focus else None
    circle=best['fit'] if best and best['fit'] else fit_2d_circle(xy[target,0],xy[target,1],d[target])
    intersections=[]
    fig=plt.figure(figsize=(12,9));ax=fig.add_subplot(111,projection='3d')
    shown=set(np.flatnonzero(truth==tid))|set(best['confirmed'] if best else [])
    for i in sorted(shown):
        ends=geo[i,:2].tolist()+[0.];ends=np.asarray(ends)[None,:]+np.array([-.5,.5])[:,None]*geo[i,5]*geo[i,2:5]
        color=('#d62d2d' if truth[i]!=tid else '#159947' if best and i in best['confirmed'] else '#d78000') if stereo[i] else '#2268aa';ax.plot(*ends.T,color=color,alpha=.6,lw=.8)
        if stereo[i] and circle:
            for offset in [0.,float(d[i]),-float(d[i])]:
                for point in finite_cylinder_intersections(circle,geo[i],offset):
                    intersections.append(dict(node=int(i),hit_id=int(hit[i]),cell_id=int(cells[i]),**point))
                    ax.scatter(point['x'],point['y'],point['z'],s=12,c='#159947' if offset==0 else '#cb3166')
    if circle:
        curve=arc(circle,xy[target]);z=np.linspace(-800,800,2)
        xx,yy=curve.T
        ax.plot_surface(np.tile(xx,(2,1)),np.tile(yy,(2,1)),np.tile(z[:,None],(1,len(xx))),alpha=.08,color='cyan')
    ax.set(xlabel='x / mm',ylabel='y / mm',zlabel='z / mm',title=f'Evt{eid} / T{tid} 几何补图：蓝=直丝；橙=目标斜丝；绿线=确认斜丝；红线=串入其他真轨斜丝\n绿点=名义圆柱交点，粉点=R±d交点（均为候选，不是真实hit）')
    ax.set_title(ax.get_title()+('\n圆柱来源：输出组的诊断重拟合' if best else '\n圆柱来源：真值分组离线参照，不是模型重建轨迹'),fontsize=10)
    ax.set_box_aspect((1,1,2));fig.tight_layout();fig.savefig(OUT/(stem+'_geometry.png'),dpi=130);plt.close(fig)
    packet=dict(event_id=eid,truth_id=tid,model=model,model_label=MODELS[model],weight_sha256=weight,source_legacy=str(ROOT.parent/'AI工程化4 - 副本/STEP6_事件切片可视化.py'),category=cause,strict=strict,relaxed=relaxed,matched=tid in matched,missed_axial_truth_ids=sorted(missed),missing_3d_only_truth_ids=sorted(missing_3d_only),annotation_version=2,groups=groups,diagnostic_truth_group_rms=reference_rms,target_axial_hits=len(target),target_distinct_cells=len(np.unique(cells[target])),geometry_circle=circle,geometry_circle_source='最相交重建组的诊断圆' if best else '真值分组仅作离线参照',intersections=intersections)
    e.save_json(OUT/(stem+'.json'),packet)
    membership={i:[] for i in range(len(cells))}
    for g in groups:
        for i in g['indices']:membership[i].append(f'C{g["number"]}')
        for i in g['confirmed']:membership[i].append(f'C{g["number"]}:confirmedS')
    with (OUT/(stem+'_hits.csv')).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['node','HitID','CellID','TrkID','is_stereo','d_mm','reference_x','reference_y','A_group_or_confirmed_S'])
        w.writerows([i,int(hit[i]),int(cells[i]),int(truth[i]),bool(stereo[i]),float(d[i]),*geo[i,:2],';'.join(membership[i])] for i in range(len(cells)))
    with (OUT/(stem+'_edges.csv')).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['node_u','node_v','HitID_u','HitID_v','type','probability','same_truth'])
        w.writerows([int(a),int(b),int(hit[a]),int(hit[b]),('S' if stereo[a] else 'A')+('S' if stereo[b] else 'A'),float(prob),bool(label)] for (a,b),prob,label in zip(pairs.T,p,true) if truth[a]==tid or truth[b]==tid)
    gallery();print(json.dumps(dict(image=str(path),model=model,event=eid,truth=tid)),flush=True);return path


def gallery():
    cards=[]
    for p in sorted(OUT.glob('event_*.json')):
        x=read(p);stem=p.stem
        cards.append(f'<article><h2>Evt{x["event_id"]} / T{x["truth_id"]} · {html.escape(x["model_label"])} · {html.escape(x["category"])}</h2><a href="{stem}.png"><img src="{stem}.png"></a><p><a href="{stem}_geometry.png">三维几何补图</a> · <a href="{stem}.json">圆参数/组信息</a> · <a href="{stem}_hits.csv">全部hit归属</a> · <a href="{stem}_edges.csv">目标相关AA/AS/SS边</a></p></article>')
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>AI6 六联图诊断</title><style>body{font:16px sans-serif;background:#eef1f4;margin:24px}article{background:white;padding:16px;margin:24px 0}img{width:100%}a{color:#1768ae}</style><h1>AI6 六联图诊断</h1><p>继承旧版2×3：上排全事件，下排局部；左=真值分组观测，中=模型AA边，右=重建。真值仅离线诊断。斜丝几何另图，不把丝中心当真实hit。比较的是固定epoch126及11月9日微调快照。</p>'+''.join(cards),encoding='utf-8')


def gui():
    import tkinter as tk
    from tkinter import ttk,messagebox
    from PIL import Image,ImageTk
    import os
    app=tk.Tk();app.title('AI6 六联图诊断 · 旧版2×3布局');width=min(1450,app.winfo_screenwidth()-60);height=min(1000,app.winfo_screenheight()-90);app.geometry(f'{width}x{height}')
    bar=ttk.Frame(app);bar.pack(fill='x',padx=8,pady=8)
    category=tk.StringVar(value='全部');ttk.Label(bar,text='原漏轨类别').pack(side='left')
    box=ttk.Combobox(bar,textvariable=category,values=['全部']+list(LABELS.values()),state='readonly',width=25);box.pack(side='left')
    model=tk.StringVar(value=MODELS['baseline']);ttk.Label(bar,text='模型').pack(side='left',padx=8)
    ttk.Combobox(bar,textvariable=model,values=list(MODELS.values()),state='readonly',width=21).pack(side='left')
    ev=tk.StringVar();tr=tk.StringVar();ttk.Label(bar,text='EvtID').pack(side='left');ttk.Entry(bar,textvariable=ev,width=8).pack(side='left');ttk.Label(bar,text='TrkID').pack(side='left');ttk.Entry(bar,textvariable=tr,width=8).pack(side='left')
    msg=tk.StringVar(value='双击病例生成六联图；首次新模型需重建该事件，完成后自动显示。点击图像看原尺寸。')
    ttk.Label(app,textvariable=msg).pack(fill='x',padx=8)
    options=ttk.Frame(app);options.pack(fill='x',padx=8)
    strict=tk.StringVar(value='.70');relaxed=tk.StringVar(value='.575')
    ttk.Label(options,text='连边显示：严格').pack(side='left');ttk.Entry(options,textvariable=strict,width=7).pack(side='left')
    ttk.Label(options,text='宽松').pack(side='left');ttk.Entry(options,textvariable=relaxed,width=7).pack(side='left')
    ttk.Label(options,text='仅改变连边着色，重建分组仍为原阈值结果。固定200验证事件 / 223问题病例；其他事件可能无概率缓存。').pack(side='left',padx=8)
    tree=ttk.Treeview(app,columns=('event','truth','category','hits'),show='headings',height=6)
    for k,label in zip(tree['columns'],['EvtID','TrkID','原问题类别','直丝hit数']):tree.heading(k,text=label)
    tree.pack(fill='x',padx=8)
    image_label=ttk.Label(app);image_label.pack(expand=True,fill='both');state={};cases=catalog()
    def populate(*_):
        for i in tree.get_children():tree.delete(i)
        for c in cases:
            if category.get()=='全部' or category.get()==LABELS[c['category']]:tree.insert('', 'end',values=(c['event'],c['truth'],LABELS[c['category']],c['hits']))
    def select(*_):
        if tree.selection():
            values=tree.item(tree.selection()[0])['values'];ev.set(values[0]);tr.set(values[1])
    def generate(*_):
        if state.get('proc') and state['proc'].poll() is None:return
        try:eid=int(ev.get());tid=int(tr.get())
        except ValueError:messagebox.showerror('输入','请选择病例或输入整数EvtID/TrkID');return
        try:
            s=float(strict.get());l=float(relaxed.get())
            if not 0<l<=s<1:raise ValueError()
        except ValueError:messagebox.showerror('阈值','要求0 < 宽松 ≤ 严格 < 1');return
        key=next(k for k,v in MODELS.items() if v==model.get())
        OUT.mkdir(exist_ok=True);log=(OUT/'gui.log').open('w',encoding='utf-8');state['log']=log
        state['path']=OUT/f'event_{eid:05d}_T{tid}_{key}.png'
        state['proc']=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'--event',str(eid),'--track',str(tid),'--model',key,'--strict-threshold',str(s),'--relaxed-threshold',str(l)],stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        msg.set('正在生成六联图与几何补图……');poll()
    def poll():
        if state['proc'].poll() is None:app.after(500,poll);return
        state['log'].close()
        if state['proc'].returncode:msg.set('生成失败，查看 visual_diagnosis_20261121/gui.log');return
        im=Image.open(state['path']);im.thumbnail((width-40,height-260));state['img']=ImageTk.PhotoImage(im);image_label.configure(image=state['img']);msg.set('已生成；点击图片查看原尺寸，图集按钮可查看三维补图和CSV。')
    ttk.Button(bar,text='生成六联图',command=generate).pack(side='left',padx=8)
    ttk.Button(bar,text='打开图集/补图',command=lambda:os.startfile(OUT/'index.html')).pack(side='left')
    image_label.bind('<Button-1>',lambda _:os.startfile(state['path']) if 'path' in state and state['path'].exists() else None)
    tree.bind('<<TreeviewSelect>>',select);tree.bind('<Double-1>',generate);box.bind('<<ComboboxSelected>>',populate);populate()
    available=sorted(OUT.glob('event_*.png'))
    if available:
        preferred=OUT/'event_06978_T1456_baseline.png'
        state['path']=preferred if preferred.exists() else next((p for p in available if not p.stem.endswith('_geometry')),available[0]);im=Image.open(state['path']);im.thumbnail((width-40,height-260));state['img']=ImageTk.PhotoImage(im);image_label.configure(image=state['img'])
        packet=read(state['path'].with_suffix('.json'));ev.set(packet['event_id']);tr.set(packet['truth_id'])
        for item in tree.get_children():
            values=tree.item(item)['values']
            if int(values[0])==packet['event_id'] and int(values[1])==packet['truth_id']:tree.selection_set(item);tree.see(item);break
    app.mainloop()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--gui',action='store_true');parser.add_argument('--examples',action='store_true');parser.add_argument('--event',type=int);parser.add_argument('--track',type=int);parser.add_argument('--model',choices=MODELS,default='baseline');parser.add_argument('--strict-threshold',type=float,default=.7);parser.add_argument('--relaxed-threshold',type=float,default=.575);args=parser.parse_args()
    if not 0<args.relaxed_threshold<=args.strict_threshold<1:parser.error('要求0 < 宽松 ≤ 严格 < 1')
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    import torch
    torch.set_num_threads(1)
    if args.gui or len(sys.argv)==1:gui();return
    if args.examples:
        for kind in LABELS:
            c=next(c for c in catalog() if c['category']==kind);draw(c['event'],c['truth'],args.model,args.strict_threshold,args.relaxed_threshold)
    elif args.event is not None and args.track is not None:draw(args.event,args.track,args.model,args.strict_threshold,args.relaxed_threshold)
    else:parser.error('使用--gui，或--examples，或--event ID --track ID')


if __name__=='__main__':main()
