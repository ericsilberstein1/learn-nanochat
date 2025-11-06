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

memory_allocated_on_previous_log = 0

def log_memory_stats(message, relevant_tensors = {}, indentation = 0):
    if not os.environ.get("LOG_MEMORY_STATS"):
        return
    allocated = torch.cuda.memory_allocated()
    global memory_allocated_on_previous_log
    delta = allocated - memory_allocated_on_previous_log
    memory_allocated_on_previous_log = allocated
    print(f"{' ' * indentation}{message} - now allocated: {allocated / 1024**3:.3f} GiB, delta: {delta / 1024**2:.3f} MiB")
    for name, tensor in relevant_tensors.items():
        print(f"{' ' * indentation}> {name} - {list(tensor.shape)} - {tensor.dtype} - {tensor.untyped_storage().nbytes() / 1024**2:.3f} MiB")

