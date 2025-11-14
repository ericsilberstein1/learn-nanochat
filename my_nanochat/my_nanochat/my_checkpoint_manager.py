# "copied" from https://github.com/karpathy/nanochat/blob/master/nanochat/checkpoint_manager.py

import os
import torch
import json
from my_nanochat.my_common import print0
from my_nanochat.my_gpt import GPTConfig, GPT
from my_nanochat.my_tokenizer import get_tokenizer

def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        torch.save(model_data, model_path)
        print(f"saved model to {model_path}")
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
        print(f"saved metadata to {meta_path}")
    if optimizer_data is not None:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        torch.save(optimizer_data, optimizer_path)
        print(f"saved optimizer to {optimizer_path}")

def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)
    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        optimizer_data = torch.load(optimizer_path, map_location=device)
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta_data = json.load(f)
    return model_data, optimizer_data, meta_data

def build_model(checkpoint_dir, step, device, phase):
    assert phase in ['train', 'eval']
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, device)
    if device.type in {'cpu', 'mps'}:
        # convert bfloat16 to float
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v for k, v in model_data.items()
        }
    # Hack: fix torch compile issue, which prepends all keys with _orig_mod.
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    model_config_kwargs = meta_data["model_config"]
    print0(f"Building model with config: {model_config_kwargs}")
    model_config = GPTConfig(**model_config_kwargs)
    with torch.device("meta"):
        model = GPT(model_config)
    model.to_empty(device=device)
    model.init_weights() # his note: this is dumb, but we need to init the rotary embeddings. TODO: fix model re-init
    model.load_state_dict(model_data, strict=True, assign=True) # keys must match, properties of tensors from model_data are preserved
    if phase == "eval":
        model.eval()
    else:
        model.train()
    tokenizer = get_tokenizer()
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"]
    return model, tokenizer, meta_data
