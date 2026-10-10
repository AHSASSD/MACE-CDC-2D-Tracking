# AI5 v2 Edge Graph

This isolated iteration adds a local edge-graph experiment on top of the frozen
AI4 time model and the existing AI5 node-state classifier.

The experiment is deliberately non-destructive:

- AI4 first-pass tracks remain unchanged.
- Shared nodes are never used as the sole fit anchor.
- A shared node may be reused only when two residual edge candidates form a
  compatible local split (a triplet-like relation).
- New candidates are accepted only after exclusive-node circle fitting and the
  existing geometry arbitration.
- Truth labels are used only by the offline report evaluator.

Run from this directory or pass absolute paths:

```powershell
python `
  "iterations\v2_edge_graph\STEP4_局部边图共享分叉.py" `
  --audit-dir "data\audit_validation_3000" `
  --model-file "../AI工程化4/best_interaction_network_time_clean_physics_v2.pth" `
  --node-model-file "models\best_node_state_classifier_node_state_v1.pth" `
  --rounds 300 `
  --output "reports\edge_graph_v2_local_in_validation_300.json"
```

## First 300-graph result

`edge_graph_v2_validation_300.json` is the initial rule-based prototype. It
produced zero supplemental candidates because the current condition required
two already-formed CA/Hough fragments to touch the same shared articulation.
That condition is too narrow for Cell-competition losses: the missing track is
often not represented by a second classical fragment at all.

The next implementation step is therefore a genuine local rerun seeded by a
shared articulation: construct the induced local edge graph from the original
Interaction Network edge probabilities, classify local articulation/elbow
relations, and only then form a short-track candidate. The original AI4 and
classic hybrid branches remain the locked controls.

The triplet classifier uses multiclass focal loss (`gamma=2.5`) with clipped
inverse-frequency alpha weights so the shared-split and contaminated classes
receive more training emphasis than the dominant continuation class.
