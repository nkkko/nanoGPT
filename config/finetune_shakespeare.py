import time
import torch

# Force float16 instead of bfloat16
dtype = 'float16'
torch._dynamo.config.suppress_errors = True  # Add this to prevent compilation errors

# env var to help with memory fragmentation
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

out_dir = '/tmp/out-shakespeare'
eval_interval = 5
eval_iters = 40
wandb_log = True
wandb_project = 'shakespeare'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'shakespeare'
init_from = 'gpt2-medium' # You can try other GPT-2 model medium, large, xl

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 1
gradient_accumulation_steps = 16
max_iters = 20

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False

# Add gradient checkpointing
gradient_checkpointing = True

# Reduce eval_iters to save memory during evaluation
eval_iters = 20

# Only save best checkpoints
always_save_checkpoint = False