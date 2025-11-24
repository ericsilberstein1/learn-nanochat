import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from collections import deque
from contextlib import nullcontext
import time
import torch
import torch.distributed as dist
import wandb


from my_nanochat.my_common import get_base_dir, autodetect_device_type, compute_init, print0, DummyWandb, compute_cleanup
from my_tasks.my_common import MyTaskMixture
from my_tasks.my_smoltalk import MySmolTalk
from my_tasks.my_mmlu import MyMMLU
from my_tasks.my_gsm8k import MyGSM8K
from my_tasks.my_customjson import MyCustomJSON
from my_tasks.my_spellingbee import MySimpleSpelling, MySpellingBee
from my_nanochat.my_tokenizer import get_tokenizer
from my_nanochat.my_checkpoint_manager import load_model, save_checkpoint
from my_nanochat.my_tokenizer import get_token_bytes
from my_nanochat.my_loss_eval import evaluate_bpb
from my_nanochat.my_report import get_report



# config
run = "dummy"
device_type = ""
model_tag = None
step = None # step to load model from
dtype = "bfloat16"
num_iterations = -1
max_seq_len = 2048
device_batch_size = 32
unembedding_lr = 0.004
embedding_lr = 0.2
matrix_lr = 0.02
init_lr_frac = 1.0
weight_decay = 0.0
eval_every = 150
eval_tokens = 20 * 32 * 2048 * 8
total_batch_size = 32 * 2048 * 8
dry_run = 0
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'my_nanochat', 'my_configurator.py')).read())
user_config = {k: globals()[k] for k in config_keys}
print0(f"user_config: {user_config}")
#---

# compute init
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process =  ddp_rank == 0
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None  # wait for GPU to finish operations
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# wandb logging init
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project='my-nanochat-mid', name=run, config=user_config)

# load model and tokenizer
model, tokenizer, meta_data = load_model('base', device, phase='train', model_tag=model_tag, step=step)
pretrain_batch_size = meta_data.get('device_batch_size', None)
if pretrain_batch_size is not None and device_batch_size > pretrain_batch_size:
    print0(f"WARNING: base model model training used device_batch_size {pretrain_batch_size} but here device_batch_size is {device_batch_size}")
orig_model = model
model = torch.compile(model, dynamic=False)
depth = model.config.n_layer
# TODO num_flops_per_token
tokens_per_fwdbwd = device_batch_size * max_seq_len # for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")
token_bytes = get_token_bytes(device=device)

# optimizers
optimizers = model.setup_optimizers(
    unembedding_lr=unembedding_lr,
    embedding_lr=embedding_lr,
    matrix_lr=matrix_lr,
    weight_decay=weight_decay,
)
adamw_optimizer, muon_optimizer = optimizers
for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["lr"] * init_lr_frac
        group["initial_lr"] = group["lr"]

# data
base_dir = get_base_dir()
identity_conversations_filepath = os.path.join(base_dir, "identity_conversations.jsonl")
train_dataset = MyTaskMixture([
    MySmolTalk(split="train"), # 460K rows of general conversations
    MyMMLU(subset="auxiliary_train", split="train"), # 100K rows of multiple choice problems drawn from ARC, MC_TEST, OBQA, RACE
    MyGSM8K(subset="main", split="train"), # 8K rows teaching simple math and (calculator) tool use
    MyCustomJSON(filepath=identity_conversations_filepath), # 1000 rows of synthetic identity conversations
    MyCustomJSON(filepath=identity_conversations_filepath), # let's do 2 epochs of these
    MySimpleSpelling(size=200000, split="train"), # 200K rows of Simple Spelling (e.g. spell the word 'apple')
    MySpellingBee(size=80000, split="train"), # 80K rows of Spelling Bee (e.g. how many 'r' are in 'strawberry'?)
]) # total: 460K + 100K + 8K + 200K + 80K = 848K rows
val_dataset = MyTaskMixture([
    MySmolTalk(split="test"), # 24K rows in test set
    MyMMLU(subset="all", split="test", stop=5200), # 14K rows in test set, use only 5.2K to match the train ratios
    MyGSM8K(subset="main", split="test", stop=420), # 1.32K rows in test set, use only 420 to match the train ratios
]) # total: 24K + 14K + 1.32K ~= 39K rows

last_step = False
approx_progress = 0.0
def mid_data_generator(split):
    global last_step, approx_progress
    assert split in ['train', 'val']
    dataset = train_dataset if split == 'train' else val_dataset
    dataset_size = len(dataset)
    assert dataset_size > 0
    needed_tokens = device_batch_size * max_seq_len + 1
    token_buffer = deque()
    scratch = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=(device_type == 'cuda'))
    cursor = ddp_rank
    it = 0
    while True:
        while len(token_buffer) < needed_tokens:
            conversation = dataset[cursor]
            ids, _ = tokenizer.render_conversation(conversation) # why don't we care about the mask?
            token_buffer.extend(ids)
            cursor += ddp_world_size
            if cursor >= dataset_size:
                cursor -= dataset_size
                if split == "train":
                    last_step = True
        it += 1
        if num_iterations > 0 and it >= num_iterations and split == 'train':
            last_step = True
        for i in range(needed_tokens):
            scratch[i] = token_buffer.popleft()
        inputs_cpu = scratch[:-1].to(dtype=torch.int32)
        targets_cpu = scratch[1:]
        inputs = inputs_cpu.view(device_batch_size, max_seq_len).to(device=device, dtype=torch.int32, non_blocking=True)
        targets = targets_cpu.view(device_batch_size, max_seq_len).to(device=device, dtype=torch.int64, non_blocking=True)
        if split == 'train':
            if num_iterations > 0:
                approx_progress = it / num_iterations
            else:
                approx_progress = cursor / dataset_size
        yield inputs, targets

train_loader = mid_data_generator('train')
build_val_loader = lambda: mid_data_generator('val')
progress = 0

def get_lr_multiplier(progress):
    return 1 if progress < 0.8 else 1 - (progress - 0.8) / 0.2

def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# training loop
x, y = next(train_loader)
min_val_bpb = float('inf')
smooth_train_loss = 0
ema_beta = 0.9
total_training_time = 0
step = 0
while True:
    # TODO flops_so_far

    if ddp:
        last_step_tensor = torch.tensor(last_step, dtype=torch.int32, device=device)
        dist.all_reduce(last_step_tensor, op=dist.ReduceOp.MAX)
        last_step = bool(last_step_tensor.item())

    if eval_every > 0 and (last_step or step % eval_every == 0):
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
    
    if master_process and last_step and not dry_run:
        # save checkpoint
        output_dirname = f"d{depth}" # why not support model_tag?
        checkpoint_dir = os.path.join(get_base_dir(), "mid_checkpoints", output_dirname)
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            [opt.state_dict() for opt in optimizers],
            {
                "step": step,
                "val_bpb": val_bpb,
                "model_config": {
                    "sequence_len": max_seq_len,
                    "vocab_size": tokenizer.get_vocab_size(),
                    "n_layer": depth,
                    "n_head": model.config.n_head,
                    "n_kv_head": model.config.n_kv_head,
                    "n_embd": model.config.n_embd,
                },
                "user_config": user_config,
                "device_batch_size": device_batch_size,
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
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y = next(train_loader)
        progress = max(progress, approx_progress)
    
    lrm = get_lr_multiplier(progress)
    for opt in optimizers:
        for group in opt.param_groups:
            group['lr'] = group['initial_lr'] * lrm
    muon_momentum = get_muon_momentum(step)
    for group in muon_optimizer.param_groups:
        group["momentum"] = muon_momentum
    for opt in optimizers:
        opt.step()

    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0

    step += 1

    # logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(total_batch_size / dt)
    # TODO flops_per_sec
    # TODO promised_flops_per_sec
    # TODO mfu
    mfu = -1 # TODO
    if step > 10:
        total_training_time += dt

    print0(f"step {step:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | total time: {total_training_time/60:.2f}m")
    if step % 10 == 0:
        wandb_run.log({
            'step': step,
            'total_training_time': total_training_time,
            'train/loss': debiased_smooth_loss,
            'train/lrm': lrm,
            'train/dt': dt,
            'train/tok_per_sec': tok_per_sec,
        })
        # TODO log flops and mfu once have

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

if not dry_run:
    get_report().log(section='Midtraining', data=[
        user_config,
        {
            "Number of iterations": step,
            "DDP world size": ddp_world_size,
        },
        {
            "Minimum validation bpb": min_val_bpb,
        }
    ])


# cleanup
wandb_run.finish()
compute_cleanup()
