# standard transformer w/ sparse TopK MLPs (streaming from openwebtext)

run_name = "main"
out_dir = "out/main"
tb_log_dir = "out/main/tb"
data_dir = "data/tinystories"

# model
n_layer = 5
n_head = 8
n_embd = 256
block_size = 256

# sparsity
use_relu_before_topk = True
tie_weights = False

# grid search hyperparameters
mlp_mult_values = [1, 2, 4, 8, 16, 32, 64]
topk_values = [64, 32, 16, 8]
with_no_topk = True  # Also train with K=None (no TopK constraint)

# HuggingFace Hub upload settings
hf_repo_id = None  # set to your own HF repo (with HF_TOKEN) to upload checkpoints instead of saving them to out_dir
hf_token = None  # read from the HF_TOKEN env var

# training
device = "cuda"
init_seed = 1234
data_seed = 5678

batch_size = 64
gradient_accumulation_steps = 1  # accumulate gradients over N steps before updating
learning_rate = 3e-4
min_lr = 1e-5
warmup_iters = 0
lr_decay_iters = 300000
max_iters = 300000


weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

eval_interval = 1000
eval_iters = 200
log_interval = 10

always_save_checkpoint = True
save_all_checkpoints = False
keep_last_checkpoints = 3


# checkpoint_fractions = None
checkpoint_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  
save_initial_checkpoint = True
save_final_checkpoint = True

amp = True

# dead neuron tracking thresholds (activation frequency per token)
dead_freq_thresholds = [1e-6]

resample_dead_neurons = True
resample_dead_threshold = 5e-6
resample_max_per_layer = 0  # 0 => unlimited
resample_interval = 1  # resample every K eval intervals (1 => every eval, 2 => every other eval, etc.)


