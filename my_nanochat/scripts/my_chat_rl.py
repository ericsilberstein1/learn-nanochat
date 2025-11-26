import os
import itertools
import re
from contextlib import nullcontext
import wandb
import torch
import torch.distributed as dist

from my_nanochat.my_common import compute_init, compute_cleanup, print0, get_base_dir, DummyWandb, autodetect_device_type
from my_nanochat.my_checkpoint_manager import save_checkpoint, load_model
from my_nanochat.my_engine import Engine
from my_tasks.my_gsm8k import MyGSM8K
from my_nanochat.my_report import get_report


# hyperparameters config
run = "dummy"
model_tag = None
source = "sft"
dtype = 'bfloat16'
device_type = ""
num_steps = -1  # so can test on mac
device_batch_size = 8
examples_per_step = 16
num_samples = 16
max_new_tokens = 256
temperature = 1.0
top_k = 50
unembedding_lr = 0.004
embedding_lr = 0.2
matrix_lr = 0.02
weight_decay = 0.0
init_lr_frac = 0.05
num_epochs = 1
save_every = 60
eval_every = 60
eval_examples = 400

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'my_nanochat', 'my_configurator.py')).read())
user_config = {k: globals()[k] for k in config_keys}
print0(f"user_config: {user_config}")

# compute init
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process =  ddp_rank == 0
ptdtype = torch.float32 if dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

# wandb logging init
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project='my-nanochat-rl', name=run, config=user_config)

# load model and tokenizer
model, tokenizer, meta_data = load_model(source, device, phase='eval', model_tag=model_tag)
engine = Engine(model, tokenizer)

train_task = MyGSM8K(subset="main", split="train")
val_task = MyGSM8K(subset="main", split="test")
num_steps = (len(train_task) // examples_per_step) * num_epochs if num_steps == -1 else num_steps
print0(f"Calculated number of steps: {num_steps}")

@torch.no_grad()
def get_batch():
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    rank_indices = range(ddp_rank, len(train_task), ddp_world_size)
    for example_idx in itertools.cycle(rank_indices):
        conversation = train_task[example_idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_len = len(tokens)

        model.eval() # this is pretty different, we're going to use the model in generating a batch
        generated_token_sequences = []
        masks = []
        num_sampling_steps = num_samples // device_batch_size
        for sampling_step in range(num_sampling_steps):
            seed = hash((step, example_idx, sampling_step)) & 0x7FFFFFFF
            with autocast_ctx:
                generated_token_sequences_batch, masks_batch = engine.generate_batch(
                    tokens,
                    num_samples=device_batch_size,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    seed=seed,
                )
            generated_token_sequences.extend(generated_token_sequences_batch)
            masks.extend(masks_batch)
    
        rewards = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_len:]
            generated_text = tokenizer.decode(generated_tokens)
            reward = train_task.reward(conversation, generated_text) # 1 or 0 right?
            rewards.append(reward)

        max_len = max(len(seq) for seq in generated_token_sequences)
        padded_generated_token_sequences = [seq + [assistant_end] * (max_len - len(seq)) for seq in generated_token_sequences]
        padded_masks = [mask + [0] * (max_len - len(mask)) for mask in masks]

        ids = torch.tensor(padded_generated_token_sequences, dtype=torch.long, device=device)
        mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)

        inputs = ids[:, :-1]
        targets = ids[:, 1:].clone()
        targets[mask_ids[:, 1:] == 0] = -1
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)

        mu = rewards.mean()
        advantages = rewards - mu

        yield generated_token_sequences, inputs, targets, rewards, advantages

def run_gsm8k_eval(task, tokenizer, engine,
                   max_examples=None,
                   num_samples=1,
                   max_completion_tokens=256,
                   temperature=0.0,
                   top_k=50):
    max_examples = min(max_examples, len(task)) if max_examples is not None else len(tasks)
    for idx in range(ddp_rank, max_examples, ddp_world_size):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)
        assert num_samples <= device_batch_size # he comments can add loop if not, won't be true on mac
        generated_token_sequences, masks = engine.generate_batch(tokens,
                                                                 num_samples=num_samples,
                                                                 max_tokens=max_completion_tokens,
                                                                 temperature=temperature,
                                                                 top_k=top_k)
        outcomes = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            is_correct = task.evaluate(conversation, generated_text)
            outcomes.append({
                'is_correct': is_correct
            })

        record = {
            'idx': idx,
            'outcomes': outcomes,
        }
        yield record


# optimizers
optimizers = model.setup_optimizers(
    unembedding_lr=unembedding_lr,
    embedding_lr=embedding_lr,
    matrix_lr=matrix_lr,
    weight_decay=weight_decay,
)
for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["lr"] * init_lr_frac
        group["initial_lr"] = group["lr"]

# lr scheduler
def get_lr_multiplier(it):
    lrm = 1.0 - it / num_steps
    return lrm

print0(f"total sequences per step: {examples_per_step * num_samples}")
assert examples_per_step % ddp_world_size == 0
examples_per_rank = examples_per_step // ddp_world_size
print0(f"calculated examples per rank: {examples_per_rank}")

batch_iterator = get_batch()
for step in range(num_steps):

    if step % eval_every == 0:
        # evaluate model
        model.eval()
        passk = torch.zeros(device_batch_size, device=device)
        with autocast_ctx:
            records_iter = run_gsm8k_eval(
                val_task,
                tokenizer,
                engine,
                num_samples=device_batch_size,
                max_examples=eval_examples,
                temperature=1.0)
            records = list(records_iter)
        for k in range(1, device_batch_size + 1):
            passk[k-1] = sum(any(o["is_correct"] for o in r["outcomes"][:k]) for r in records)
        num_records = torch.tensor(len(records), dtype=torch.long, device=device)
        if ddp:
            dist.all_reduce(num_records, op=dist.ReduceOp.SUM)
            dist.all_reduce(passk, op=dist.ReduceOp.SUM)
        passk = passk / num_records.item()
        print_passk = [f"Pass@{k}: {passk[k - 1].item():.4f}" for k in range(1, device_batch_size + 1)]
        print0(f"Step {step} | {', '.join(print_passk)}")
        log_passk = {f"pass@{k}": passk[k - 1].item() for k in range(1, device_batch_size + 1)}
        wandb_run.log({
            "step": step,
            **log_passk,
        })

    
    # forward / backwards
    rewards_list = []
    sequence_lengths = []
    for example_step in range(examples_per_rank):
        sequences_all, inputs_all, targets_all, rewards_all, advantages_all = next(batch_iterator)
        model.train()
        assert inputs_all.size(0) % device_batch_size == 0
        num_passes = inputs_all.size(0) // device_batch_size
        for pass_idx in range(num_passes):
            b0, b1 = pass_idx * device_batch_size, (pass_idx + 1) * device_batch_size
            inputs = inputs_all[b0:b1]
            targets = targets_all[b0:b1]
            rewards = rewards_all[b0:b1]
            advantages = advantages_all[b0:b1]
            with autocast_ctx:
                logp = -model.forward(inputs, targets, loss_reduction='none').view_as(inputs) # (B, T)
            pg_obj = (logp * advantages.unsqueeze(-1)).sum()
            num_valid = (targets >= 0).sum().clamp(min=1)
            pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)
            loss = -pg_obj
            loss.backward()
            print0(f"Step {step}/{num_steps} | Example step {example_step} | Pass {pass_idx} | loss: {loss.item():.6f} | average reward: {rewards.mean().item()}")

        # for logging
        rewards_list.append(rewards_all.mean().item())
        sequence_lengths.extend(len(seq) for seq in sequences_all)

    # bunch of logging for how the rollouts went this step
    mean_reward = sum(rewards_list) / len(rewards_list)
    mean_sequence_length = sum(sequence_lengths) / len(sequence_lengths)
    if ddp:
        mean_reward_tensor = torch.tensor(mean_reward, dtype=torch.float, device=device)
        mean_sequence_length_tensor = torch.tensor(mean_sequence_length, dtype=torch.float, device=device)
        dist.all_reduce(mean_reward_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(mean_sequence_length_tensor, op=dist.ReduceOp.AVG)
        mean_reward = mean_reward_tensor.item()
        mean_sequence_length = mean_sequence_length_tensor.item()
    print0(f"Step {step}/{num_steps} | Average reward: {mean_reward} | Average sequence length: {mean_sequence_length:.2f}")
    wandb_run.log({
        'step': step,
        'reward': mean_reward,
        'sequence_length': mean_sequence_length,
    })

    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group['lr'] = group['initial_lr'] * lrm
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    wandb_run.log({
        'step': step,
        'lrm': lrm,
    })

    if master_process and ((step > 0 and step % save_every == 0) or step == num_steps - 1):
        base_dir = get_base_dir()
        depth = model.config.n_layer
        model_tag = f"d{depth}" # why not support model_tag?
        checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints", model_tag)
        model_config_kwargs = model.config.__dict__
        save_checkpoint(
            checkpoint_dir,
            step,
            model.state_dict(),
            None, # optimizer
            {
                'model_config': model_config_kwargs
            }
        )
        print(f"Saved model checkpoint to {checkpoint_dir}")

get_report().log(section='Chat RL', data=[
    user_config,
])

wandb_run.finish()
compute_cleanup()
