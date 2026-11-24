"""Cross-check saved diagnostics and add a compact comparison without rerunning models."""
import csv,json
from pathlib import Path
import reconstructibility as r


def main():
    out=r.OUT;contract=r.read(out/'contract.json');summary=r.read(out/'summary.json')
    rows=[t for eid in contract['event_ids'] for t in r.read(out/'survey'/('event_%05d.json'%eid))['tracks']]
    original=r.read(r.BATCH/'summary.json')
    assert len(rows)==original['metrics']['truth_tracks']
    assert sum(x['found'] for x in rows)==original['metrics']['matched_tracks']
    for x in rows:
        c=x['largest_true_component']
        assert c['all']>=c['relaxed']>=c['strict']>=1
        assert 0<=x['exact_input_equivalent_fraction']<=x['shared_cell_fraction']<=1
    events=[r.read(out/'details'/('event_%05d.json'%eid)) for eid in r.read(out/'selected_details.json')]
    requested=[];comparison=[]
    for event in events:
        focus=[x for x in event['focus'] if not x['found']]
        for x in focus:
            for flag,mode in [('oracle_remove_false_found','remove_false_edges'),('oracle_denoised_nonlocking_found','remove_false_and_nonlocking_single_pass'),
                              ('oracle_perfect_graph_found','perfect_candidate_edge_labels'),('oracle_3d_cell_matching_found','oracle_3d_cell_matching')]:
                assert x[flag]==(x['truth_id'] in event['oracle_trials'][mode]['found'])
        if event['event_id'] not in [1500,1501,1503,1504]:continue
        requested.extend(focus);t=event['oracle_trials']['oracle_3d_cell_matching']['counts']
        comparison.append(dict(event_id=event['event_id'],missed=len(focus),
            remove_false=sum(x['oracle_remove_false_found'] for x in focus),
            remove_false_nonlocking=sum(x['oracle_denoised_nonlocking_found'] for x in focus),
            perfect_edges=sum(x['oracle_perfect_graph_found'] for x in focus),
            oracle_circle_templates=sum(x['oracle_circle_assignment_found'] for x in focus),
            oracle_3d_templates=sum(x['oracle_3d_assignment_found'] for x in focus),
            oracle_3d_cell_matching=sum(x['oracle_3d_cell_matching_found'] for x in focus),
            baseline_matched=event['baseline_counts']['matched_tracks'],
            conditional_matched=t['matched_tracks'],conditional_fake=t.get('fake_candidates',0),
            conditional_clone=t.get('clone_candidates',0),truth_tracks=event['baseline_counts']['truth_tracks'],
            cell_assumption_violations=event['oracle_trials']['oracle_3d_cell_matching']['truth_tracks_violating_assumption']))
    reasons=dict(false_edges=sum(x['oracle_remove_false_found'] for x in requested),
        additional_nonlocking=sum(not x['oracle_remove_false_found'] and x['oracle_denoised_nonlocking_found'] for x in requested),
        needs_score_change=sum(not x['oracle_denoised_nonlocking_found'] and x['oracle_perfect_graph_found'] for x in requested),
        still_failed_perfect_edges=sum(not x['oracle_perfect_graph_found'] for x in requested))
    assert sum(reasons.values())==len(requested)
    summary.update(requested_by_event=comparison,requested_intervention_attribution=reasons)
    r.e.save_json(out/'summary.json',summary)
    with (out/'representative_comparison.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(comparison[0]));w.writeheader();w.writerows(comparison)
    marker='## 四个指定事例的归因补充'
    text=(out/'诊断报告.md').read_text(encoding='utf-8').split(marker)[0].rstrip()
    lines=['',marker,'','以下“找回”只计原来漏掉的直丝轨；三维模板和丝内竞争是已知真值模板后的条件实验。',
        '', '|事件|原漏轨|仅去假边|去假边并取消锁点|理想连边|三维模板+丝内竞争|该条件下整事件匹配/真轨|假轨/克隆|',
        '|---|---|---|---|---|---|---|---|']
    for x in comparison:
        lines.append('|%d|%d|%d|%d|%d|%d|%d/%d|%d/%d|'%(x['event_id'],x['missed'],x['remove_false'],x['remove_false_nonlocking'],x['perfect_edges'],x['oracle_3d_cell_matching'],x['conditional_matched'],x['truth_tracks'],x['conditional_fake'],x['conditional_clone']))
    lines+=['',f"四个事例共{len(requested)}条原漏轨：仅去假边恢复{reasons['false_edges']}条；取消提前锁点再恢复{reasons['additional_nonlocking']}条；其余{reasons['needs_score_change']}条需要改善评分/连通性。", 
        '', '事件1504在真值模板条件下，经有限丝几何及丝内一对一竞争可匹配31/31条直丝轨，假轨和克隆均为0；这说明截图中的7条漏轨值得保留为难例。不过1500、1501、1503仍有归属混淆，不能将1504的条件结果推广到全部重叠事例。',
        '', '这四个事例未发现单轨重复命中同一CellID，符合本次丝内一对一条件。多圈轨、重复穿越同一根丝时不能直接使用此限制。',
        '', '扫描中没有发现跨真轨且CellID与float32模型输入漂移距离完全相同的直丝hit；这只排除了最直接的局部特征完全重复，不证明噪声下的全局唯一可辨识性。']
    (out/'诊断报告.md').write_text(text+'\n'+'\n'.join(lines)+'\n',encoding='utf-8')
    result=dict(status='passed',survey_events=len(contract['event_ids']),tracks=len(rows),representatives=comparison,attribution=reasons,
                source_sha256=r.e.sha(Path(__file__)),analysis_contract_sha256=r.e.sha(out/'contract.json'))
    r.e.save_json(out/'verification.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
