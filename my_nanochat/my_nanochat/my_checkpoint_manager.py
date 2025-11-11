# "copied" from https://github.com/karpathy/nanochat/blob/master/nanochat/checkpoint_manager.py

import os
import torch
import json

def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data):
    assert int(os.environ.get('RANK', 0)) == 0
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    torch.save(model_data, model_path)
    print(f"saved model to {model_path}")
    if optimizer_data is not None:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}.pt")
        torch.save(optimizer_data, optimizer_path)
        print(f"saved optimizer to {model_path}")
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=2)
    print(f"saved metadata to {meta_path}")

