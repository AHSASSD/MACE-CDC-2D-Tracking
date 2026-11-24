"""Archive the user-supplied paper and record a migration assessment; no training."""
from pathlib import Path
import hashlib,json,shutil

ROOT=Path(__file__).resolve().parent
SOURCE=Path(r'E:\wechat_files\xwechat_files\wxid_xgulu2358izw22_aeb5\temp\RWTemp\2026-05\ac23f8d5092fab20d7d51a7fd9b8460c\LHCb-EXT4VELO-2406.12869v1(1).pdf')
PROJECT=ROOT.parent.parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    folder=ROOT/'references';folder.mkdir(exist_ok=True)
    target=folder/'2406.12869v1_ETX4VELO.pdf'
    if target.exists():assert sha(target)==sha(SOURCE)
    else:shutil.copy2(SOURCE,target)
    budget=json.loads((ROOT/'triplet_budget.json').read_text(encoding='utf-8'))
    sizes=[x['potential_triplets'] for x in budget['rows'] if x['edge_threshold']==.4]
    original=PROJECT.parent/'AI工程化6_稳定基线备份/AI6_稳定基线_20261107_230240/AI工程化6/common.py'
    record=dict(paper=dict(title='Graph Neural Network-Based Pipeline for Track Finding in the Velo at LHCb',
        arxiv='2406.12869',supplied_version='v1',archived_pdf=str(target),pdf_sha256=sha(target),
        all11_pages_read=True,figures_5_6_visually_checked=True,
        key_sections='3.3-3.5, PDF pages6-7, printed pages5-6',
        primary_urls=['https://arxiv.org/html/2406.12869v1','https://arxiv.org/html/2406.12869v3'],
        website_version_checked='v3, revised2025-11-25',
        version_comparison_scope='Sections3.3-3.5 retain the triplet types, shared encoder classification and track-construction assumptions.',
        official_code='https://gitlab.cern.ch/gdl4hep/etx4velo/-/tree/ctd2023',official_code_inspected=False),
      sketch_mapping=[dict(user_type='I',paper_type='Articulation',edges=['A->C','C->B']),
                      dict(user_type='II',paper_type='Left elbow',edges=['C->A','C->B']),
                      dict(user_type='III',paper_type='Right elbow',edges=['A->C','B->C'])],
      correction='Types describe pairs of edges at a common hit, not exclusive classes assigned by total node degree. Short-tracklet compression is optional, not a prerequisite for a triplet classifier.',
      method=dict(edge_graph='Original edges become vertices; compatible edge pairs become links.',
          supervision='A small triplet DNN reuses GNN hit/edge encodings; paper trains edge and triplet objectives jointly.',
          reconstruction='Merge genuine elbow groups, exclude fork articulations from WCC, process remaining fork links and map edge groups back to hits.',
          caveat='Elbow type is not itself a fake label; each topology still needs genuine/fake classification.'),
      transfer_requirements=[
          'Define local direction/order for curved or returning drift-chamber tracks; forward z/plane order from VELO is unavailable.',
          'Use drift radii and finite wire geometry for axial/stereo consistency; wire reference coordinates are not measured3D hit positions.',
          'Preserve distinct HitIDs sharing CellID and use the original exclusive ownership/matching conventions when scoring.',
          'Test repeated overlap/merge/diverge cases: the paper assumes a shared sequence diverges only once.',
          'Expose model intermediate states: current baseline returns logits only; its node state is12D and transient edge effects are256D.',
          'Use bounded/chunked triplet generation and classification. Potential triplet budget here is before directional/geometry cuts.',
          'Use original A>=5, purity>=.8, completeness>=.5 plus coarse3D seed criteria; paper70% matching is a different metric.',
          'Resolve target/score conventions against official code before copying the printed focal-loss equation.'],
      budget=dict(events=[1500,1501,1503,1504],edge_threshold=.4,
                  minimum_potential_triplets=min(sizes),maximum_potential_triplets=max(sizes),
                  definition='sum over hits of degree*(degree-1)/2 for unique undirected retained edges, including AA/AS/SS'),
      assessment='Concrete published precedent makes the triplet/edge-graph route worth a bounded prototype. Performance transfer is unproven.',
      proposed_validation_order=['Audit orientation, candidate coverage and triplet budget.',
          'Use truth triplets only as an offline upper-bound check.',
          'Freeze the existing GNN and evaluate a small learned/geometric triplet head using training/validation splits.',
          'Compare full tracking efficiency, fake/clone rates and runtime before any joint GNN retraining.'],
      original_model_source_sha256=sha(original),current_training_changed=False,tracker_deployment_changed=False,
      source_sha256=sha(Path(__file__)))
    path=ROOT/'paper_review.json';path.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(pdf=str(target),review=str(path),paper_sha256=record['paper']['pdf_sha256'],triplets_at_0p4=record['budget']),ensure_ascii=False))


if __name__=='__main__':main()
