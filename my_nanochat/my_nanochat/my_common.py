import os
import torch
import torch.distributed as dist

def print0():
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)

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

def is_ddp():
    # TODO
    return False

def compute_init(device_type="cuda"):
    assert device_type in ["cuda", "mps", "cpu"]
    if device_type == "cuda":
        assert torch.cuda.is_available()
    if device_type == "mps":
        assert torch.backends.mps.is_available()
    
    torch.manual_seed(42)
    if device_type == "cuda":
        torch.cuda.manual_seed(42)
    
    if device_type == "cuda":
        torch.set_float32_matmul_precision('high') # he notes this ues tf32 instead of fp32 for matmuls

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if ddp and device_type == "cuda":
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device) # make cuda default to this device
        dist.init_process_group(backend="nccl", device_id=device) # NCCL = NVIDIA Collective Communications Library 
        dist.barrier() # this is a synchronization primitive, do you only call it once?
    else:
        device = torch.device(device_type)

    # TODO if ddp_rank == 0 log

    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device

def compute_cleanup():
    if is_ddp():
        dist.destroy_process_group()


# for challenge-15-understand-memory-needed/understand-memory-needed.ipynb
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

