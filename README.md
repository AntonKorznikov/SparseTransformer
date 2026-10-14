# Sparse Transformer: a Controlled Testbed for Mechanistic Interpretability

Code for the paper *"Sparse Transformer: a Controlled Testbed for Mechanistic Interpretability"*.

The **Sparse Transformer (ST)** is a 5-layer decoder-only language model trained on
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) whose MLPs are wide
($N = M \cdot d_\text{model}$ neurons) and exactly $K$-sparse:

```
MLP(x) = V_out · TopK(ReLU(V_in x))
```

so every MLP output is a sum of $K$ known value vectors. This gives a known reference basis for
evaluating interpretability methods (SAEs, Transcoders, circuit discovery) on natural language.

- **Weights:** all 35 models of the $(M, K)$ grid, each with 11 checkpoints (every 10% of training):
  [huggingface.co/AntonKorznikov/SparseTransformer](https://huggingface.co/AntonKorznikov/SparseTransformer)
- **Neuron dashboards** for all 40,960 neurons of the primary model (M=32, K=8):
  [huggingface.co/datasets/AntonKorznikov/SparseTransformer-dashboards](https://huggingface.co/datasets/AntonKorznikov/SparseTransformer-dashboards)
- **Paper:** arXiv link coming soon.

## Setup

```bash
git clone https://github.com/AntonKorznikov/SparseTransformer.git
cd SparseTransformer
pip install -r requirements.txt

# tokenize TinyStories with the GPT-2 tokenizer -> data/tinystories/{train,val}.bin, meta.pkl
python data/tinystories/prepare.py
```

All commands below are run **from the repository root**.

### Download weights

```bash
# primary model (M=32, K=8), all 11 checkpoints -> out/main_mlp_mult_32_K_8/
python download_from_hf.py --folder main_mlp_mult_32_K_8

# only the final checkpoint
python download_from_hf.py --folder main_mlp_mult_32_K_8 --pattern ckpt_iter300000.pt

# dense baselines used in the paper
python download_from_hf.py --folder main_mlp_mult_1_K_None --pattern ckpt_iter300000.pt
python download_from_hf.py --folder main_mlp_mult_4_K_None --pattern ckpt_iter300000.pt
python download_from_hf.py --folder main_mlp_mult_32_K_None --pattern ckpt_iter300000.pt
```

### Load a model

```python
import torch
from huggingface_hub import hf_hub_download
from model_transformer import SparseTransformerLM, TransformerConfig

path = hf_hub_download("AntonKorznikov/SparseTransformer", "main_mlp_mult_32_K_8/ckpt_iter300000.pt")
ckpt = torch.load(path, map_location="cpu", weights_only=False)
model = SparseTransformerLM(TransformerConfig(**ckpt["model_cfg"]))
model.load_state_dict(ckpt["model"])
model.eval()

# neuron i of layer l:  key k_i = blocks[l].mlp.fc1.weight[i],  value v_i = blocks[l].mlp.fc2.weight[:, i]
```

## Reproducing the paper

| Paper section | Code | Command |
|---|---|---|
| §3.2 Training the 35-model grid | `train_transformer.py`, `config/main_config.py` | `python train_transformer.py config/main_config.py` |
| §3.3 Width–sparsity trade-off, dead neurons (Fig. 2, Tab. 1) | `scaling/plot_eval.ipynb`, `scaling/eval_models.py` | run the notebook (uses the included `scaling/recomputed_metrics.pkl`); `python -m scaling.eval_models` recomputes the metrics |
| App. C Neuron dashboards | `dashboards/create_sae_dashboard.py`, `dashboards/create_html_dashboard.py`, `dashboards/export_neuron_dashboards_pngs.py` | see below |
| §4.2, App. B Circuits (Fig. 1, Tab. 4–5) | `circuits/create_attribution_graph.py`, `circuits/generate_circuit_*_table.py` | see below |
| §5, App. A Case Study #1: SAE vs Transcoder (Fig. 3, Tab. 3) | `case_study_1/distill_mlp.py` and runners, `case_study_1/make_results_table.py`, `case_study_1/distill_suite_tensorboard_article_plots.ipynb` | see below |
| §6 Case Study #2: polysemanticity under compression (Fig. 4–5) | `case_study_2/*.py` | see below |

### Training

`config/main_config.py` trains the full grid $M \in \{1,2,4,8,16,32,64\}$, $K \in \{8,16,32,64,\text{None}\}$
sequentially (300k steps each, batch 64, context 256, AdamW, cosine LR $3\cdot10^{-4} \to 10^{-5}$,
dead-neuron resampling). To train a single model, edit `mlp_mult_values` / `topk_values` in the config.
Checkpoints are saved to `out/main/` (or uploaded to your own HF repo if you set `hf_repo_id` in the config and `HF_TOKEN`); TensorBoard logs go to `out/main/tb`.

### Width–sparsity trade-off

`scaling/recomputed_metrics.pkl` contains the validation loss, dead-neuron fraction (activation frequency
< 1e-6, averaged over the 5 layers) and active neurons per token for all 35 models, as used in the paper.
`scaling/plot_eval.ipynb` draws the figures, fits the power law $\mathcal{L}(M,K) = c\,M^{\alpha}K^{\beta}$
(log-space least squares; for K=None the measured number of active neurons per token is used as K),
computes the singular values of the loss matrix and compares linear / logarithmic alternatives.

`python -m scaling.eval_models` recomputes these metrics from the final HF checkpoints and writes
`scaling/recomputed_metrics_rerun.pkl` (35 models, takes a few hours on one GPU).

### Neuron dashboards

Precomputed dashboards for every neuron of `main_mlp_mult_32_K_8` are available at
[AntonKorznikov/SparseTransformer-dashboards](https://huggingface.co/datasets/AntonKorznikov/SparseTransformer-dashboards).
To regenerate them:

```bash
# collect top-activating contexts for all neurons and render HTML dashboards
python -m dashboards.create_sae_dashboard --ckpt out/main_mlp_mult_32_K_8/ckpt_iter300000.pt \
  --out_dir out/main_mlp_mult_32_K_8/neuron_dashboards
# (re)render HTML from saved data
python -m dashboards.create_html_dashboard --data_file <layerX_dashboard_data.pkl> --output_dir <dir>
# PNG snapshots of random neurons (needs `pip install pyppeteer`)
python -m dashboards.export_neuron_dashboards_pngs out/main_mlp_mult_32_K_8/neuron_dashboards --samples-per-layer 4
```

### Circuits

```bash
# one prompt (~20 min on an A100)
python -m circuits.create_attribution_graph --search_mode beam --max_nodes 50 \
  --prompt "Jack was hungry, so he went looking for something to"
# all 23 default prompts (the paper reports 16 of them; ~7 h)
python -m circuits.create_attribution_graph --search_mode beam --max_nodes 50
# per-prompt metrics table (Tab. 4) and F/C-vs-size table (Tab. 5) for the 16 paper prompts
python -m circuits.generate_circuit_curve_tables
python -m circuits.generate_circuit_metrics_table
```

Candidates are the active (post-TopK) MLP neurons at all positions; they are ranked by single-neuron
zero-ablation of P(target), and beam search (width 40) maximises faithfulness F for every circuit size
n ≤ 50. Completeness C is evaluated on the same circuits.

Outputs (interactive HTML graphs, JSON curves) go to `out/main_mlp_mult_32_K_8/attribution_graphs/`.
See `circuits/CIRCUIT_DETAIL_USAGE.md` for the detailed-circuit options.

### Case Study #1: SAE vs Transcoder

All experiments use layer 2 of the M=32, K=8 model; the checkpoint is downloaded from HF automatically.

```bash
# main comparison (D = 8192, K = 8): collects teacher MLP input/output pairs, trains Transcoder + SAE,
# and the dense student used in Case Study #2 (hidden size 1024, no TopK)
python -m case_study_1.run_all_distill_experiments --save_target_final_ckpts --only_target_models
# results table (Tab. 3) from the TensorBoard logs
python -m case_study_1.make_results_table
```

NMSE and MMCS against the true keys/values are logged to TensorBoard under `out/distill_suite/`;
`case_study_1/distill_suite_tensorboard_article_plots.ipynb` draws Fig. 3.
The ± in the table is the spread of the validation NMSE over the last evaluations of the run.
`pb0` is the SAE without biases (reported in the paper), `pb1` the SAE with a
shared pre-bias.

### Case Study #2: polysemanticity under compression

Requires the dense student trained by `run_all_distill_experiments` above (hidden size 1024 despite the
`h10240` folder name).

```bash
python -m case_study_2.case_study_2 \
  --teacher_ckpt out/main_mlp_mult_32_K_8/ckpt_iter300000.pt \
  --student_ckpt out/distill_suite/target_models/results_real_h10240_knone/h1024_knone_relu1_final.pt
```

This produces the teacher–student concentration curves, the PMI-vs-TVD histogram and the key/value
alignment histograms in `figures/`. The `case_study_2/analyze_*.py` scripts are the individual analyses
(conditional mapping $a(s_j \mid t_i)$, $b(t_i \mid s_j)$, pairwise metrics vs TVD, geometric alignment)
and take the same `--teacher_ckpt` / `--student_ckpt` arguments.

## Notes on the training code

`train_transformer.py` is the exact code the released models were trained with, including two quirks
that are kept so that the code matches the weights:

- dead-neuron resampling resets the `fc1` row and `fc2` column of a dead neuron but not its `fc1` bias;
- for K=None models every neuron is counted as active during training, so they are never resampled.

Both MLP linear layers have biases, so an MLP output is a sum of at most K value vectors plus the `fc2` bias.

## Repository layout

```
model_transformer.py   Sparse Transformer (RoPE, RMSNorm, TopK(ReLU) MLP)
model_adapter.py       hooked model wrapper used by dashboards and circuits
train_transformer.py   grid training with neuron resampling and HF upload
config/                training and distillation configs
utils/common.py        data loading helpers
data/tinystories/      dataset preparation
scaling/  dashboards/  circuits/  case_study_1/  case_study_2/
```

## Citation

```bibtex
@article{korznikov2026sparsetransformer,
  title  = {Sparse Transformer: a Controlled Testbed for Mechanistic Interpretability},
  author = {Korznikov, Anton and others},
  year   = {2026}
}
```

## License

MIT
