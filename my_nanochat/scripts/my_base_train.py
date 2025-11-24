# copying from https://github.com/karpathy/nanochat/blob/master/scripts/base_train.py

# explored some of this code earlier in challenge-13-baby-pretrain/baby-pretrain.ipynb

# working on this from challenge-17-implement-base-train-script/implement-base-train-script.ipynb

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from my_nanochat.my_common import autodetect_device_type, compute_init, compute_cleanup, print0, get_base_dir, DummyWandb
from my_nanochat.my_tokenizer import get_tokenizer, get_token_bytes
from my_nanochat.my_gpt import GPTConfig, GPT
from my_nanochat.my_dataloader import tokenizing_distributed_data_loader
from my_nanochat.my_checkpoint_manager import save_checkpoint
from my_nanochat.my_loss_eval import evaluate_bpb
from my_nanochat.my_engine import Engine
from scripts.my_base_eval import evaluate_model
from my_nanochat.my_report import get_report

import wandb

run = "dummy" # wandb run name, "dummy" means don't log

device_type = ""

# model architecture
depth = 20
max_seq_len = 2048

# training horizon
num_iterations = -1
# TODO target_flops
target_param_data_ratio = 20 # calculate num_iterations to maintain fixed data:param ratio (Chinchilla=20) (-1 = disable)

# optimization
device_batch_size = 32
total_batch_size = 524288
embedding_lr = 0.2
unembedding_lr = 0.004
weight_decay = 0.0
matrix_lr = 0.02
grad_clip = 1.0
warmup_ratio = 0.0
warmdown_ratio = 0.2
final_lr_frac = 0.0

#evaluation
eval_every = 250
eval_tokens = 20*524288
core_metric_every = 2000
core_metric_max_per_task = 500
sample_every = 2000

model_tag = ""

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'my_nanochat', 'my_configurator.py')).read())
user_config = {k: globals()[k] for k in config_keys}

print0(f"user_config: {user_config}")

device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
print(f"This process is ddp_rank: {ddp_rank}, ddp_local_rank: {ddp_local_rank}, ddp_world_size: {ddp_world_size}")
master_process = ddp_rank == 0
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None  # wait for GPU to finish operations
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# wandb logging init
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project='my-nanochat', name=run, config=user_config)

tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# model kwargs
num_layers = depth
model_dim = depth * 64 # see his note about this
num_heads = max(1, (model_dim + 127) // 128)
num_kv_heads = num_heads # GQA of 1:1, same as GQA being disabled
print0(f"num_layers: {num_layers}")
print0(f"model_dim: {model_dim}")
print0(f"num_heads: {num_heads}")
print0(f"num_kv_heads: {num_kv_heads}")

# optimizer / data / training length related hyperparameters
# figure out needed gradient accumulation to reach the desired total batch size
tokens_per_fwdbwd = device_batch_size * max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# init model
model_config_kwargs = dict(
    sequence_len=max_seq_len,
    vocab_size=vocab_size, 
    n_layer=num_layers,
    n_head=num_heads,
    n_kv_head=num_kv_heads,
    n_embd=model_dim,
)
with torch.device("meta"):
    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config)

print0(model)
model.to_empty(device=device)
model.init_weights()
orig_model = model
model = torch.compile(model, dynamic=False)
num_params = sum(p.numel() for p in model.parameters())
print0(f"Number of parameters: {num_params:,}")
# TODO num_flops_per_token = model.estimate_flops()
# print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

if num_iterations > 0:
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif target_param_data_ratio > 0:
    target_tokens = target_param_data_ratio * num_params
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated num iterations from target_param_data_ratio as {num_iterations:,}")
# TODO training horizon based on target_flops

total_tokens = total_batch_size * num_iterations
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"tokens : param ratio: {total_tokens / num_params:.2f} (he has note that Chinchilla is ~20)")
# TODO print total training FLOPs estimate

# initialize optimizer
optimizers = model.setup_optimizers(
    unembedding_lr=unembedding_lr,
    embedding_lr=embedding_lr,
    matrix_lr=matrix_lr,
    weight_decay=weight_decay,
)
adamw_optimizer, muon_optimizer = optimizers

# initialize DataLoaders
train_loader = tokenizing_distributed_data_loader(device_batch_size, max_seq_len, split="train", device=device)
build_val_loader = lambda: tokenizing_distributed_data_loader(device_batch_size, max_seq_len, split="val", device=device)
x, y = next(train_loader)
print0(f"x.shape is {list(x.shape)}, y.shape is {list(y.shape)} -- should match")

# set up hyperparameter scheulders
# learning rate scheduler
def get_lr_multiplier(it):
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac

def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85  + frac * 0.95
    return momentum

# training loop
min_val_bpb = float("inf")
smooth_train_loss = 0 # EMA of training loss
ema_beta = 0.9 # EMA decay factor
total_training_time = 0 # wall-clock time
# run +1 steps to eval and save at end
for step in range(num_iterations+1):
    last_step = step == num_iterations
    # TODO flops_so_far = num_flops_per_token * total_batch_size * step
    
    if last_step or step % eval_every == 0:
        # once in a while evaluate the val bpb
        model.eval()
        val_loader = build_val_loader()
        eval_steps = eval_tokens // (device_batch_size * max_seq_len * ddp_world_size)
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"step {step:05d} | Validation bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            'step': step,
            'total_training_time': total_training_time,
            'val/bpb': val_bpb,
        })
        # TODO log flops once have
        model.train()

    if core_metric_every > 0 and (last_step or (step > 0 and step % core_metric_every == 0)):
        model.eval()
        with autocast_ctx:
            results = evaluate_model(orig_model, tokenizer, device, max_per_task=core_metric_max_per_task)
        print0(f"Step {step:05d}: CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            'step': step,
            'core_metric': results['core_metric'],
            'centered_results': results['centered_results'],
        })
        # TODO log flops once have
        model.train()

    if master_process and (last_step or (step > 0 and step % sample_every == 0)):
        # once in a while sample from the model
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer)
        for prompt in prompts:
            tokens = tokenizer.encode(prompt, prepend=tokenizer.get_bos_token_id())
            with autocast_ctx:
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=10, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    if master_process and last_step:
        # save checkpoint
        output_dirname = model_tag if model_tag else f"d{depth}"
        checkpoint_dir = os.path.join(get_base_dir(), "base_checkpoints", output_dirname)
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            [opt.state_dict() for opt in optimizers],
            {
                "step": step,
                "val_bpb": val_bpb,
                "model_config": model_config_kwargs,
                "user_config": user_config,
                "device_batch_size": device_batch_size,
                "max_seq_len": max_seq_len,
            }
        )

    if last_step:
        break

    # single training step
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach() # for logging
        loss = loss / grad_accum_steps # make sure understand his comment: each .backward() is a grad sum => normalize loss here
        loss.backward()
        x, y = next(train_loader)
    # gradient clipping
    grad_clip_enabled = grad_clip > 0
    if grad_clip_enabled:
        # TODO better understand grad clipping
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(orig_model.parameters(), grad_clip) # does this return the norm before clipping?
        grad_norm = grad_norm_tensor.item() # GPU tensor -> CPU float (note cpu-gpu sync point)
    # step the optimizers
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    muon_momentum = get_muon_momentum(step)
    for group in muon_optimizer.param_groups:
        group["momentum"] = muon_momentum
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0

    # logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    # TODO flops_per_sec
    # TODO promised_flops_per_sec
    mfu = -1 # TODO mfu
    if step > 10:
        total_training_time += dt
    print_grad_norm = f" grad norm: {grad_norm:.4f} |" if grad_clip_enabled else ""
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} |{print_grad_norm} lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | total time: {total_training_time/60:.2f}m")
    if step % 100 == 0:
        log_data = {
            'step': step,
            'total_training_time': total_training_time,
            'train/loss': debiased_smooth_loss,
            'train/lrm': lrm,
            'train/dt': dt,
            'train/tok_per_sec': tok_per_sec,
        }
        # TODO add total_training_flops and mfu once have
        if grad_clip_enabled:
            log_data['train/grad_norm'] = grad_norm
        wandb_run.log(log_data)

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

# log to report
get_report().log(section='Base model training', data=[
    user_config,
    { # stats about the training setup
        "Number of parameters": num_params,
        # TODO "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Calculated number of iterations": num_iterations,
        "Number of training tokens": total_tokens,
        "Tokens : Params ratio": total_batch_size * num_iterations / num_params,
        "DDP world size": ddp_world_size,
        "warmup_ratio": warmup_ratio,
        "warmdown_ratio": warmdown_ratio,
        "final_lr_frac": final_lr_frac,
    },
    { # stats about training outcomes
        "Minimum validation bpb": min_val_bpb,
        "Final validation bpb": val_bpb,
        "CORE metric estimate": results.get("core_metric", None),
        # TODO "MFU %": f"{mfu:.2f}%",
        # TODO "Total training flops": f"{flops_so_far:e}",
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
    }])

# cleanup
wandb_run.finish()
compute_cleanup()



