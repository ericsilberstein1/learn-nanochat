import os
from contextlib import nullcontext
import torch
from my_nanochat.my_checkpoint_manager import load_model
from my_nanochat.my_common import compute_init, print0, compute_cleanup, autodetect_device_type
from my_nanochat.my_dataloader import tokenizing_distributed_data_loader
from my_nanochat.my_tokenizer import get_token_bytes
from my_nanochat.my_loss_eval import evaluate_bpb
from my_nanochat.my_engine import Engine
from my_nanochat.my_report import get_report

# config
device_batch_size = 32
split_tokens = 20 * 524288
model_tag = None
model_step = None
device_type = ""
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'my_nanochat', 'my_configurator.py')).read())
user_config = {k: globals()[k] for k in config_keys}

print0(f"user_config: {user_config}")

# Load the base model and the tokenizer
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=model_tag, step=model_step)
sequence_len = meta["model_config"]["sequence_len"] # could be arbitrary really
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

tokens_per_step = device_batch_size * sequence_len * ddp_world_size
assert split_tokens % tokens_per_step == 0
steps = split_tokens // tokens_per_step
token_bytes = get_token_bytes(device=device)
bpb_results = {}
for split_name in ['train', 'val']:
    loader = tokenizing_distributed_data_loader(device_batch_size, sequence_len, split_name, device=device)
    with autocast_ctx:
        bpb = evaluate_bpb(model, loader, steps, token_bytes)
    print0(f"{split_name} bpb: {bpb:.4f}")
    bpb_results[split_name] = bpb

samples = []
if ddp_rank == 0:
    prompts = [
        "The capital of France is",
        "The chemical symbol of gold is",
        "If yesterday was Friday, then tomorrow will be",
        "The opposite of hot is",
        "The planets of the solar system are:",
        "My favorite color is",
        "If 5*x + 3 = 13, then x is",
    ]
    engine = Engine(model, tokenizer)
    for prompt in prompts:
        tokens = tokenizer.encode(prompt, prepend=tokenizer.get_bos_token_id())
        with autocast_ctx:
            sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
        sample_str = tokenizer.decode(sample[0])
        print0(sample_str)
        samples.append(sample_str)

get_report().log(section='Base model loss', data=[
    {
        'train bpb': bpb_results['train'],
        'val bpb': bpb_results['val'],
    },
    {f"sample {i}": sample for i, sample in enumerate(samples)},
])

compute_cleanup()