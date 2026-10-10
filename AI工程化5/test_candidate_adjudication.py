from STEP4_shared候选审计 import apply_selected_rule, branch, passes_rule, select_rule, summarize


def row(outcome, borrowed, anchor_hits=5, rms=0.1):
    return {
        "outcome": outcome, "borrowed_shared_hits": borrowed, "visible_hits": 6,
        "anchor_hits": anchor_hits, "rms": rms, "mean_shared_residual": 0.1,
        "mean_edge_probability": 0.8, "min_edge_probability": 0.6,
        "mean_shared_probability": 0.99, "min_shared_probability": 0.98,
        "mean_uncertainty": 0.05, "mean_owner_residual_on_anchors": 3.0,
    }


def main():
    rows = [row("new_match", 0), row("new_match", 0), row("fake", 0, rms=0.9)]
    selected = select_rule(rows, "no_borrow")
    accepted = apply_selected_rule(rows, "no_borrow", selected)
    assert summarize(accepted)["new_match"] == 2
    assert summarize(accepted)["fake"] == 0
    assert branch(row("new_match", 1)) == "borrowed"
    assert passes_rule(row("new_match", 0), [("rms", "<=", 0.2)])
    print("candidate adjudication self-check passed")


if __name__ == "__main__":
    main()
