# grid config (used by distill_mlp.py --grid)

hidden_sizes = [256, 512, 1024, 2048, 4096, 8192]
mlp_topk_list = [None]

eps = 1e-8
max_steps = 50_000
batch_size = 4096
learning_rate = 3e-4
min_lr_ratio = 0.1
weight_decay = 0.0
beta1 = 0.9
beta2 = 0.99

log_interval = 1
eval_interval = 50


