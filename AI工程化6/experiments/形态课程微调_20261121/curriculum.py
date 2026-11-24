"""No replacement; all valid original train events restored from epoch16 onward."""
import numpy as np


def epoch_ids(selection,epoch):
    rng=np.random.default_rng(20261121+epoch)
    clean=selection['clean_event_ids'];hard=selection['hard_event_ids']
    # Retain some difficult examples from the first epoch to reduce forgetting.
    fraction=.25 if epoch<=5 else min(1.,.25+.75*(epoch-5)/10)
    selected=list(clean)+rng.permutation(hard)[:int(np.ceil(fraction*len(hard)))].tolist()
    return rng.permutation(selected).tolist()
