# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such
torch.backends.cudnn.benchmark = True

out_dir = 'out-shakespeare-char'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 2
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# Dataloader optimizations
num_workers = 4
pin_memory = True

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.3

learning_rate = 5e-4  # Slightly lower initial learning rate
warmup_iters = 200    # Increase warmup period

max_iters = 4000
lr_decay_iters = 4000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model

# device settings
device = 'cuda' # run on GPU
dtype = 'float16' # use float32 instead of the default bfloat16 cause T4 which has Compute Capability 7.5 (sm_75)
compile = True # disable model compilation