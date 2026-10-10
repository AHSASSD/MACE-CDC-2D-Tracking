# Tested Results

All metrics below use the project's locked 3000-graph validation audit unless stated otherwise. Truth labels are used only for offline evaluation.

## AI4 frozen baseline

- matched tracks: `87,251`
- conditional efficiency: `98.400%`
- end-to-end efficiency: `97.272%`
- 5-6 hit efficiency: `80.206%`
- fake tracks: `1,150` (`0.383/event`)
- clone tracks: `127` (`0.042/event`)

## AI5 focal triplet local edge-graph experiment

- matched tracks: `87,273` (`+22` vs AI4)
- matched 5-6 hit tracks: `4,460` (`+19`)
- 5-6 hit efficiency: `80.549%` (`+0.343 percentage points`)
- fake tracks: `1,240` (`+90`)
- clone tracks: `140` (`+13`)
- status: experimental, not promoted

## Interpretation

The AI5 edge-graph branch recovers additional short tracks, confirming that shared-hit local rerouting is useful. Its fake and clone increases are not yet acceptable for replacing AI4. AI4 remains the stable default; AI5 is published for reproducibility and continued research.
