"""Plan artifacts only: preserve original splits, count parameters, train nothing."""
from pathlib import Path
import json,sys
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'复杂事件归因_20261107'))
import experiment_common as e
from common import JointInteractionNetwork


def main():
    e.initialize_worker();e.verify_pins()
    metadata_path=e.DATA_ROOT/'dataset_metadata_joint_no_time.json'
    metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
    train=metadata['event_ids']['train'];validation=metadata['event_ids']['validation'];test=metadata['event_ids']['test']
    assert [len(train),len(validation),len(test)]==[7000,1500,1500]
    assert len(set(train+validation+test))==10000
    debug=[1500,1501,1503,1504,1714,1532]
    required=set(e.IDS)|set(validation).intersection(debug)
    assert len(e.IDS)==200 and required<=set(validation)
    rng=np.random.default_rng(20261122)
    remaining=sorted(set(validation)-required)
    tune=sorted(required|set(rng.permutation(remaining)[:1000-len(required)].tolist()))
    confirm=sorted(set(validation)-set(tune))
    pilot=sorted(np.random.default_rng(20261124).choice(train,size=1000,replace=False).tolist())
    audit=json.loads((ROOT.parent/'形态课程微调_20261121/difficulty_audit/events.json').read_text(encoding='utf-8'))
    assert {x['event_id'] for x in audit}==set(train)
    stress=[x['event_id'] for x in sorted(audit,key=lambda x:x['hits'],reverse=True)[:20]]
    cache_budget=set(stress)
    for eid in np.random.default_rng(20261125).permutation(pilot):
        if len(cache_budget)>=100:break
        cache_budget.add(int(eid))
    assert len(tune)==1000 and len(confirm)==500 and not set(tune)&set(confirm)
    assert set(e.IDS)<=set(tune) and not set(test)&set(tune+confirm)
    partition={eid:split for split,ids in metadata['event_ids'].items() for eid in ids}
    split=dict(dataset_sha256=e.sha(metadata_path),original_train_event_ids=train,
        pilot_train_event_ids=pilot,cache_budget_train_event_ids=sorted(cache_budget),stress_train_event_ids=stress,
        original_validation_event_ids=validation,original_test_event_ids=test,
        validation_tune_event_ids=tune,validation_confirm_event_ids=confirm,
        periodic_member_validation_event_ids=e.IDS,
        known_debug_cases=[dict(event_id=i,original_split=partition[i],pristine_holdout=False) for i in debug],
        rules=['Original split memberships unchanged.','Split at event level, never at hit/triplet level.',
               'Validation confirmation is internal experiment confirmation, not a historically unseen set.',
               'Previously inspected test cases retain regression status; fresh generalization evidence needs separate locked events.'])
    e.save_json(ROOT/'split_manifest.json',split)

    backbone=sum(p.numel() for p in JointInteractionNetwork().parameters());budgets=[]
    for name,h1,h2 in [('T-small',128,64),('T-wide',256,128)]:
        head=nn.ModuleDict(dict(node_projection=nn.Linear(12,32),edge_projection=nn.Linear(256,64),
            classifier=nn.Sequential(nn.Linear(262,h1),nn.LayerNorm(h1),nn.SiLU(),
                                     nn.Linear(h1,h2),nn.LayerNorm(h2),nn.SiLU(),nn.Linear(h2,1))))
        added=sum(p.numel() for p in head.parameters())
        budgets.append(dict(name=name,route='T',backbone_parameters=backbone,additional_parameters=added,
                            combined_parameters=backbone+added,input_dimension=262,output_dimension=1,hidden=[h1,h2],trained=False))
    for name,h1,h2 in [('E-small',128,64),('E-wide',256,128)]:
        head=nn.ModuleDict(dict(hit_encoder=nn.Sequential(nn.Linear(24,64),nn.LayerNorm(64),nn.SiLU()),
            hit_edge_encoder=nn.Sequential(nn.Linear(257,64),nn.LayerNorm(64),nn.SiLU()),
            classifier=nn.Sequential(nn.Linear(515,h1),nn.LayerNorm(h1),nn.SiLU(),
                                     nn.Linear(h1,h2),nn.LayerNorm(h2),nn.SiLU(),nn.Linear(h2,4))))
        added=sum(p.numel() for p in head.parameters())
        budgets.append(dict(name=name,route='E',backbone_parameters=backbone,additional_parameters=added,
                            combined_parameters=backbone+added,input_dimension=515,output_dimension=4,hidden=[h1,h2],trained=False))
    assert backbone==181519 and [x['additional_parameters'] for x in budgets]==[59233,117985,93316,184644]
    e.save_json(ROOT/'parameter_budget.json',dict(status='CPU parameter counts only',models=budgets,
        ordered_feature_dimensions=dict(T=dict(node_context=96,edge_context=128,raw_observables_and_scores=38),
            E=dict(endpoint_families=258,current_bundle=129,neighbor_context=128)),
        endpoint_exchange='Shared scorer for both endpoint orders; average probabilities at inference.',
        no_forward_or_training_performed=True))
    plan=dict(status='proposed',version=2,date='2026-11-22',primary_backbone='stable epoch126',
        backbone_weight_sha256=e.sha(e.SNAPSHOT/'best_interaction_network_joint_no_time.pth'),
        dataset_sha256=e.sha(metadata_path),split_manifest_sha256=e.sha(ROOT/'split_manifest.json'),
        model_budget_sha256=e.sha(ROOT/'parameter_budget.json'),document_sha256=e.sha(ROOT/'模型与测试计划.md'),
        E_document_sha256=e.sha(ROOT/'方案二_E_独立模型与测试计划.md'),
        source_sha256=e.sha(Path(__file__)),data_scope='Complete event graph for frozen backbone; independent E bundle-region and T triplet heads; event-wide assignment.',
        models=budgets,candidate_edge_threshold=.4,candidate_threshold_sensitivity=[.3,.4,.575],
        routes=dict(E=dict(origin='User second sketch: green ordinary edges, yellow shared segments, copied channels and red output paths',
                requires_triplet_head=False,grouping='CellID families retaining every original HitID',
                targets=['reject0','ordinary1','shared2','shared3plus'],labels='Distinct non-noise truth tracks supported by retained underlying candidate hit edges',
                training_sample='At most4096 bundles per event, class stratified',loss='Mean cross entropy over endpoint orders, event balanced',
                inference='E class/multiplicity followed by independent expansion and geometric endpoint pairing'),
            T=dict(origin='Triplet compatibility inspired by ETX4VELO',requires_E_head=False,
                targets=['different_track','same_track'],training_sample='At most8192 triplets per event, balanced positives/hard and random negatives',
                loss='Mean BCEWithLogits over endpoint orders, event balanced')),
        E_label_feasibility_sha256=e.sha(ROOT/'E_label_feasibility.json'),
        training=dict(pilot_train_events=1000,pilot_max_epochs=10,full_train_events=7000,
            full_max_epochs=60,min_epochs_before_early_stop=20,member_check_every_epochs=5,early_stop_patience_checks=4,
            optional_max_epochs=80,extension_condition='Continue only if validation members still improve and fake/clone remain controlled; record LR and continuation.',
            milestone_epochs=[20,40,60],max_T_triplets_per_event=8192,max_E_bundles_per_event=4096,
            positive_negative_ratio_T='1:1 when available',hard_negative_fraction_among_T_negatives=.5,
            first_models=['E-small','T-small'],wide_models_condition='Only after evidence of insufficient small-head capacity',
            optimizer='AdamW',learning_rate=1e-3,minimum_learning_rate=1e-4,weight_decay=1e-4,
            first_seed=20261122,replication_seed_if_passed=20261123,backbone_frozen=True,
            epoch_definition='All7000 training events visited once; sampled example counts differ by route, report steps and actual time.'),
        runtime=dict(triplet_evaluation_chunk=8192,cache='Bounded disk shards with frozen-backbone source/weight/data hashes',
            trainer_location='GPU server',member_evaluation_location='Local CPU parallelism where useful',
            speed_comparison='Same hardware and full pipeline, including I/O and coarse3D'),
        controls=['B0 stable pipeline','B1 lower edge threshold with old extraction',
                  'E0 rule colors with E expansion','E1 E-small with expansion','E1-no-copy same E-small without expansion','E2 E-wide',
                  'T0 rule compatibility with T decoder','T1 T-small','T2 T-wide','ET optional combination, separately credited'],
        gates=['No truth-dependent candidate grouping or input features.',
            'Candidate pruning target: at most0.2 percentage-point loss of pure-connectivity upper bound versus unpruned0.4 graph.',
            'Evaluate track efficiency, fake/clone, gained/lost tracks and coarse3D seed quality at matched conditions.',
            'Use paired event-level uncertainty; do not choose size by sampled triplet accuracy alone.',
            'Keep current independent HitID ownership; no synthetic observation duplication.'],
        prior_plan_archive='archive_v1',
        current_training_changed=False,new_gpu_job_submitted=False,new_model_training_started=False)
    assert plan['backbone_weight_sha256']=='5f3f02409399cd62efce5f254be02fa004237fb61a5b4717681cad1ad90cdcac'
    e.save_json(ROOT/'plan.json',plan)
    print(json.dumps(dict(status='plan_saved',directory=str(ROOT),models=budgets,
        split_counts=dict(train=len(train),validation_tune=len(tune),validation_confirm=len(confirm),test=len(test)),
        periodic_validation_in_tune=True,new_training_started=False),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
