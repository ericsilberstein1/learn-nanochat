import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from contextlib import nullcontext
import time
import torch
import torch.distributed as dist

import wandb

from my_nanochat.my_common import get_base_dir, autodetect_device_type, compute_init, print0, get_base_dir, DummyWandb, compute_cleanup
from my_nanochat.my_tokenizer import get_tokenizer
from my_tasks.my_arc import MyARC
from my_tasks.my_smoltalk import MySmolTalk
from my_tasks.my_customjson import MyCustomJSON
from my_tasks.my_spellingbee import MySimpleSpelling, MySpellingBee
from my_tasks.my_common import MyTaskMixture
from my_tasks.my_gsm8k import MyGSM8K
from my_nanochat.my_engine import Engine
from my_nanochat.my_checkpoint_manager import load_model, save_checkpoint
from scripts.my_chat_eval import run_chat_eval

# config
run = "dummy"
source = "mid"
model_tag = None
step = None
device_type = ""
dtype = "bfloat16"
device_batch_size = 4
num_epochs = 1
num_iterations = -1
max_data_tokens = -1 # if not -1, crudely skip any docs (conversations) longer 
target_examples_per_step = 32 # bet this is 4 (device_batch_size) * 8 (num GPUs)
unembedding_lr = 0.004
embedding_lr = 0.2
matrix_lr = 0.02
weight_decay = 0.0
init_lr_frac = 0.02
eval_every = 100
eval_steps = 100
eval_metrics_every = 200
eval_metrics_max_problems = 1024
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'my_nanochat', 'my_configurator.py')).read())
user_config = {k: globals()[k] for k in config_keys}
print0(f"user_config: {user_config}")
# ------------

# compute init
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process =  ddp_rank == 0
ptdtype = torch.float32 if dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=pdtype) if device_type == "cuda" else nullcontext()

# wandb logging init
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project='my-nanochat-sft', name=run, config=user_config)

# load model and tokenizer
model, tokenizer, meta_data = load_model(source, device, phase='train', model_tag=model_tag, step=step)
orig_model = model
# model = torch.compile(model, dynamic=False) # see his comment
engine = Engine(model, tokenizer)

# data
base_dir = get_base_dir()
identity_conversations_filepath = os.path.join(base_dir, "identity_conversations.jsonl")
train_ds = MyTaskMixture([
    MyARC(subset="ARC-Easy", split="train"), # 2.3K rows
    MyARC(subset="ARC-Challenge", split="train"), # 1.1K rows
    MyGSM8K(subset="main", split="train"), # 8K rows
    MySmolTalk(split="train", stop=10_000), # 10K rows of smoltalk
    MyCustomJSON(filepath=identity_conversations_filepath), # 1K rows of synthetic identity conversations
    MySimpleSpelling(size=300, split="train"), # 300 rows of Simple Spelling (e.g. spell the word 'apple')
    MySpellingBee(size=300, split="train"), # 300 rows of Spelling Bee (e.g. how many 'r' are in 'strawberry'?)
]) # 2.3K + 1.1K + 8K + 10K + 1K + 0.3K + 0.3K = 23K rows
val_ds = MySmolTalk(split="test") # general conversations, 24K rows (though we don't actually use all of it)

def sft_data_generator(dataset, batch_size):
    pad_token_id = tokenizer.encode_special("<|assistant_end|>")
    def collate_and_yield(batch):
        nrows = len(batch)
        ncols = max(len(ids) for ids, _ in batch) - 1
        inputs = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
        targets = torch.full((nrows, ncols), -1, dtype=torch.long)
        for i, (ids, mask) in enumerate(batch):
            n = len(ids)
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            inputs[i, :n-1] = ids_tensor[:-1]
            row_targets = ids_tensor[1:]
            mask_tensor = torch.tensor(mask[1:], dtype=torch.long)
            row_targets[mask_tensor == 0] = -1
            targets[i, :n-1] = row_targets
        inputs = inputs.to(device)
        targets = targets.to(device)
        return inputs, targets
    batch = []
    while True:
        for i in range(ddp_rank, len(dataset), ddp_world_size):
            doc = dataset[i]
            ids, mask = tokenizer.render_conversation(doc)  # what if it's too long?
            if max_data_tokens == -1 or len(ids) <= max_data_tokens:
                batch.append((ids, mask))
                if len(batch) == batch_size:
                    yield collate_and_yield(batch)
                    batch = []

examples_per_step = device_batch_size * ddp_world_size
print0(f"Target examples per step: {target_examples_per_step}")
print0(f"Device batch size: {device_batch_size}")
print0(f"Examples per step is device_batch_size * ddp_world_size: {examples_per_step}")
assert target_examples_per_step % examples_per_step == 0
grad_accum_steps = target_examples_per_step // examples_per_step
print0(f" => grad accum steps: {grad_accum_steps}")

if num_iterations == -1:
    assert num_epochs > 0
    num_iterations = (len(train_ds) // target_examples_per_step) * num_epochs
train_loader = sft_data_generator(train_ds, batch_size=device_batch_size)
build_val_loader = lambda: sft_data_generator(val_ds, batch_size=device_batch_size)

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

# lr scheduler
def get_lr_multiplier(it):
    lrm = 1.0 - it / num_iterations
    return lrm

# training loop
step = 0
train_iter = train_loader
for step in range(num_iterations):
    last_step = step == num_iterations - 1

    if last_step or step % eval_every == 0:
        # evaluate validation loss
        model.eval()
        val_iter = build_val_loader()
        losses =[]
        for _ in range(eval_steps):
            val_inputs, val_targets = next(val_iter)
            with torch.no_grad(), autocast_ctx:
                loss = model(val_inputs, val_targets)
                losses.append(loss)
        val_loss = torch.stack(losses).mean()
        if ddp:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss = val_loss.item()
        print0(f"Step {step:05d} | Validation loss: {val_loss:.6f}")
        wandb_run.log({
            'step': step,
            'val_loss': val_loss,
        })
        model.train()

    if last_step or (step > 0 and step % eval_metrics_every == 0):
        model.eval()
        metrics = {}
        with torch.no_grad(), autocast_ctx:
            metrics['mmlu_acc'] = run_chat_eval(
                'MMLU',
                model,
                tokenizer,
                engine,
                batch_size=device_batch_size*2,
                max_problems=eval_metrics_max_problems)
            metrics['arc_easy_acc'] = run_chat_eval(
                'ARC-Easy',
                model,
                tokenizer,
                engine,
                batch_size=device_batch_size*2,
                max_problems=eval_metrics_max_problems)
        metrics_str = ', '.join(f'{k}: {v:.6f}' for k, v in metrics.items())
        print0(f"Step {step:05d} | {metrics_str}")
        wandb_run.log({
            'step': step,
            **metrics,
        })
        model.train()

    if last_step:
        break

    num_tokens = torch.tensor(0, device=device) # num active tokens (non-masked)
    for micro_step in range(grad_accum_steps):
        train_inputs, train_targets = next(train_iter)
        with autocast_ctx:
            loss = model(train_inputs, train_targets)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        num_tokens += (train_targets >= 0).sum()
    if ddp:
        dist.all_reduce(num_tokens, op=dist.ReduceOp.SUM)

    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group['lr'] = group['initial_lr'] * lrm
    
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)

    train_loss_item = train_loss.item()
    num_tokens_item = num_tokens.item()
    print0(f"Step {step:05d}/{num_iterations:05d} | Training loss: {train_loss_item:.6f}| lrm: {lrm:.6f}| num_tokens: {num_tokens_item:,}")
    wandb_run.log({
        'step': step,
        'lrm': lrm,
        'train_loss': train_loss_item,
        'num_tokens': num_tokens_item
    })
    step += 1

# save
if master_process:
    base_dir = get_base_dir()
    depth = model.config.n_layer
    model_tag = f"d{depth}" # why not support model_tag?
    checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", model_tag)
    model_config_kwargs = model.config.__dict__
    save_checkpoint(
        checkpoint_dir,
        step,
        model.state_dict(),
        None, # don't bother saving optimizer state
        {
            'step': step,
            'val_loss': val_loss,
            **metrics,
            'model_config': model_config_kwargs,
        }
    )
    print(f"Saved model checkpoint to {checkpoint_dir}")

# todo logging

wandb_run.finish()
compute_cleanup()
    