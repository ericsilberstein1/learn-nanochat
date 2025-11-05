import os
import torch

def get_base_dir():
    home_dir = os.path.expanduser("~")
    cache_dir = os.path.join(home_dir, ".cache")
    nanochat_dir = os.path.join(cache_dir, "my_nanochat")
    os.makedirs(nanochat_dir, exist_ok = True)
    return nanochat_dir

# return ddp, ddp_rank, ddp_local_rank, ddp_world_size
def get_dist_info():
    # for now
    return False, 0, 0, 1

def autodetect_device_type():
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps" # Metal Performance Shaders for apple silicon and amd (?)
    else:
        device_type = "cpu"
    print(f"Autodetected device type: {device_type}")
    return device_type

def memory_stats():
    free, _ = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    return f"Torch memory allocated: {allocated / 1024**3:.2f} GiB (free = {free / 1024**3:.2f} GiB)"
