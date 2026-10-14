# Detailed Circuit Analysis - Usage Guide

This guide explains how to use the new detailed circuit visualization feature in `create_attribution_graph.py`.

## Overview

The detailed circuit feature allows you to:
1. **Select a targeted circuit** either by fixed node count (K) or by minimum nodes needed to achieve a retain_ratio threshold (T)
2. **Recalculate C values** with circuit isolation - measuring each node's isolated importance within the circuit
3. **Compute detailed influence edges** - for each node, show the top 10 nodes it influences and top 10 nodes that influence it
4. **Generate interactive HTML** - a detailed visualization of the circuit with per-node influence analysis

## Command-Line Arguments

### Enable Detailed Circuit Mode
```bash
--circuit_detail_mode {none|fixed_k|threshold_t}
```
- `none` (default): Don't generate detailed circuit HTML
- `fixed_k`: Select exactly K nodes (specified by `--circuit_detail_k`)
- `threshold_t`: Select minimum K nodes where retain_ratio >= T (specified by `--circuit_detail_t`)

### Fixed K Mode
```bash
--circuit_detail_mode fixed_k --circuit_detail_k 20
```
This selects the top 20 most important nodes (lowest C values) for detailed analysis.

### Threshold T Mode
```bash
--circuit_detail_mode threshold_t --circuit_detail_t 0.9
```
This selects the minimum number of nodes such that ablating all other nodes still retains at least 90% of the original log-probability.

## Detailed Computation Steps

### Step 1: Circuit Node Selection

**Fixed K Mode:**
- Takes the top K nodes from the ranked list (by lowest C value)
- Nodes are pre-sorted by importance

**Threshold T Mode:**
- Incrementally adds nodes from the ranked list until retain_ratio >= T
- Ensures you have a minimal sufficient circuit

### Step 2: Localized C Recalculation

For each node in the selected circuit:
- Ablate all OTHER nodes in the circuit (keep this node and all nodes outside the circuit)
- Compute how much log-probability decreases
- Calculate "localized" C value measuring isolated importance within the circuit

This differs from the global C value because it accounts for interactions with other circuit nodes.

### Step 3: Detailed Influence Edges

For each node in the circuit, compute:
- **Influences_out**: Top 10 nodes whose activations change most when this node is ablated
  - Shows what downstream neurons are affected
  - Sorted by relative change magnitude

- **Influences_in**: Top 10 nodes whose ablation causes this node to change most
  - Shows what upstream neurons feed into this one
  - Sorted by relative change magnitude

Measurements are done with the full circuit active (baseline state), so edge weights reflect realistic within-circuit interactions.

## Output Files

When detailed circuit mode is enabled, you get:

### `circuit_detail.json`
Complete data including:
- Selected circuit nodes with global and localized C values
- Per-node influence matrices
- Relative change and delta matrices
- Token strings and prompt for context

### `circuit_detail.html`
Interactive visualization featuring:
- Card-based layout, one per circuit node
- Node ID (Layer.Position.Neuron)
- Associated token for context
- Base activation value
- Top 10 influences_out (nodes affected)
- Top 10 influences_in (nodes affecting)
- Visual bar charts for relative changes

## Example Usage

### Example 1: Analyze top 15 nodes with detailed influence
```bash
python -m circuits.create_attribution_graph \
  --prompt "Lily and Ben went to the garden" \
  --circuit_detail_mode fixed_k \
  --circuit_detail_k 15 \
  --out_dir outputs/circuits
```

### Example 2: Find minimal circuit for 85% retain_ratio and analyze it
```bash
python -m circuits.create_attribution_graph \
  --prompt "Jack was thirsty, so he went looking for" \
  --circuit_detail_mode threshold_t \
  --circuit_detail_t 0.85 \
  --out_dir outputs/circuits
```

### Example 3: Full analysis with detailed circuit + main graph
```bash
python -m circuits.create_attribution_graph \
  --prompt "Lily and Ben went to the garden" \
  --selection_mode add_back \
  --search_mode beam \
  --circuit_detail_mode fixed_k \
  --circuit_detail_k 25 \
  --max_nodes 300 \
  --out_dir outputs/full_analysis
```

This will generate:
- `index.html` - Main attribution graph dashboard
- `circuit_detail.html` - Detailed circuit analysis
- `circuit_detail.json` - Raw circuit data
- `graph.json` - Main graph data

## Interpreting Results

### Node Card Layout
```
┌─ Node ID (L3.5.42) Token: " and" ─────────────┐
│ Base Act: 2.3451                              │
│                                               │
│ ↓ This node influences (10)                   │
│ L2.5.10        +0.234 ████                    │
│ L2.4.20        +0.189 ███                     │
│ L3.6.15        -0.045 █                       │
│                                               │
│ ↑ Influenced by (10)                          │
│ L2.3.8         +0.567 ██████████              │
│ L2.4.12        +0.234 ████                    │
│ L1.5.3         +0.123 ██                      │
└───────────────────────────────────────────────┘
```

**Interpreting relative change colors:**
- 🔵 Blue (+): Positive change - ablating source increases target activation
- 🔴 Red (-): Negative change - ablating source decreases target activation
- Bar width: Magnitude of relative change (up to 100%)

### Comparing Global vs Localized C

- **Global C**: Node's importance in the full model context
- **Localized C**: Node's importance when isolated within the circuit
  
If `Localized C > Global C`: Node is more important within the circuit than globally  
If `Localized C < Global C`: Node's importance is reduced by interactions within the circuit

## Tips & Best Practices

1. **Choose K wisely**: 
   - K=10-20: Highly focused, interpretable circuits
   - K=30-50: Moderate complexity, good detail
   - K=100+: Complex circuits, harder to visualize

2. **Use threshold_t for reproducibility**: 
   - Automatically finds sufficient circuits
   - Compare across different prompts systematically
   - Typically find K=20-40 for 0.8-0.9 threshold

3. **Combine with main analysis**:
   - Look at main graph for global importance ranking
   - Use detailed circuit to understand local interactions
   - Cross-reference node positions across both visualizations

4. **Performance considerations**:
   - Fixed K is faster (no iterative search)
   - Threshold T requires multiple forward passes
   - Larger K means O(K²) edge computation
   - Typical runtime for K=20: 2-5 minutes
   - Typical runtime for K=50: 10-20 minutes

## Advanced: Understanding the Math

### Localized C Computation
For each circuit node $n_i$:
$$C_{local}(n_i) = \exp(\log p(n_i \text{ active, others ablated}) - \log p_0)$$

Where all other circuit nodes are ablated (set to 0), measuring $n_i$'s isolated impact.

### Influence Edges (rel_change)
For nodes $u$ and $v$ in circuit:
$$\text{rel\_change}(u \to v) = \frac{\text{activation}(v \mid u \text{ ablated}) - \text{activation}(v)}{\text{activation}(v)}$$

Measured with full circuit active (all nodes except $u$ ablated).

## Troubleshooting

**Issue**: Detailed circuit computation is very slow
- **Solution**: Reduce K or use fixed_k mode (avoids threshold search)

**Issue**: Threshold T never reached
- **Solution**: Lower T value (e.g., 0.7 instead of 0.95)

**Issue**: All relative changes are near zero
- **Solution**: Increase num_repeats for more stable measurements with norm_match

**Issue**: HTML shows empty influence lists
- **Solution**: Most nodes have <10 significant connections, which is normal for sparse circuits
