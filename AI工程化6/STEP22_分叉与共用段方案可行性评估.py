"""Read-only feasibility audit of node/edge splitting; no tracker implementation."""
from pathlib import Path
from collections import Counter
import json,sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/重叠可重建性_20261122'))
import reconstructibility as r
import networkx as nx
import numpy as np


def audit(eid):
    cells,d,truth,hit,geo,stereo,pairs,p,packet=r.load(eid)
    aa=np.flatnonzero(~stereo);axial_edges=~stereo[pairs[0]]&~stereo[pairs[1]]
    same=(truth[pairs[0]]==truth[pairs[1]])&(truth[pairs[0]]>=0)
    counts=Counter(cells[aa].tolist());degrees=[]
    missed=set(packet['missed_axial_truth_ids'])
    for cutoff in [.7,.575,.4,.3,.2]:
        mask=axial_edges&(p>=cutoff);graph=nx.Graph();graph.add_nodes_from(aa.tolist())
        graph.add_edges_from(pairs[:,mask].T.tolist())
        degree=np.asarray([graph.degree(int(i)) for i in aa]);branch=[int(i) for i in aa if graph.degree(int(i))>=3]
        pure=sum(all(truth[j]==truth[i] for j in graph.neighbors(i)) for i in branch)
        bridges=[(a,b) for a,b in nx.bridges(graph) if graph.degree(a)>=3 and graph.degree(b)>=3]
        labels=r.labels_at(len(cells),pairs,mask&same);sufficient=[]
        for tid in missed:
            ix=np.flatnonzero((truth==tid)&~stereo);need=max(5,int(np.ceil(.5*len(ix))))
            if max(Counter(labels[ix]).values())>=need:sufficient.append(int(tid))
        degrees.append(dict(cutoff=cutoff,edges=int(mask.sum()),true_edges=int((mask&same).sum()),false_edges=int((mask&~same).sum()),
            median_degree=float(np.median(degree)),max_degree=int(degree.max()),degree3_or_more=len(branch),
            degree3_but_only_same_truth_neighbors=int(pure),
            bridge_between_branch_nodes=len(bridges),such_bridges_cross_truth=int(sum(truth[a]!=truth[b] for a,b in bridges)),
            missed_with_sufficient_pure_connectivity=sufficient))
    return dict(event_id=eid,axial_hits=len(aa),unique_axial_cells=len(counts),
        axial_hit_ids_unique=len(np.unique(hit[aa]))==len(aa),
        multi_hit_cells=sum(n>1 for n in counts.values()),
        multi_hit_cell_multiplicity=dict(sorted(Counter(counts.values()).items())),
        hits_in_multi_hit_cells=sum(n for n in counts.values() if n>1),
        missed_truth_ids=sorted(missed),graph_thresholds=degrees)


def main():
    r.initialize();r.e.verify_pins();rows=[audit(eid) for eid in [1500,1501,1503,1504]]
    # A split creates alternative routes, not new edges between disconnected components.
    result=dict(scope='Feasibility only. Original events, cache, tracker and training are unchanged.',events=rows,
        color_interpretation=dict(page1='Blue/yellow branch states are copied before red path pairing; green outer nodes remain.',
                                  page2='Green ordinary edges remain, yellow potentially shared segment gets multiple route states, red paths are the output.'),
        structural_limits=['Raw candidate/high-score graphs contain dense within-track edges; degree3 is not a branch label.',
                           'Candidate scores are averaged over two directions; II versus III needs a local path orientation.',
                           'Parallel copies of an undirected edge do not specify which outer branches should be paired.',
                           'State splitting alone cannot repair a true path already disconnected by the score threshold.',
                           'A virtual copy must retain original HitID and final ownership; existing independent hits sharing CellID already exist.'],
        source_sha256=r.e.sha(Path(__file__)),dataset_sha256=r.e.sha(r.e.DATA_ROOT/'dataset_metadata_joint_no_time.json'),
        weight_sha256=r.e.CONTRACT['weight_sha256'],deployment_changed=False,training_changed=False)
    out=ROOT/'experiments/分叉方案可行性_20261122';out.mkdir(exist_ok=True);r.e.save_json(out/'audit.json',result)
    for row in rows:
        print(json.dumps(dict(event_id=row['event_id'],axial_hits=row['axial_hits'],multi_hit_cell_multiplicity=row['multi_hit_cell_multiplicity'],
                             thresholds=row['graph_thresholds']),ensure_ascii=False),flush=True)
    print(str(out/'audit.json'),flush=True)


if __name__=='__main__':main()
