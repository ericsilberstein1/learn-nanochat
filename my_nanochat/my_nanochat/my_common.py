import os
import torch
import torch.distributed as dist
from filelock import FileLock
import urllib.request

def print0(s, **kwargs):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)

def get_base_dir():
    if os.environ.get("NANOCHAT_BASE_DIR"):
        nanochat_dir = os.environ.get("NANOCHAT_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanochat_dir = os.path.join(cache_dir, "my_nanochat")
    os.makedirs(nanochat_dir, exist_ok = True)
    return nanochat_dir

# return ddp, ddp_rank, ddp_local_rank, ddp_world_size
def get_dist_info():
    if is_ddp():
        assert all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
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
    return int(os.environ.get('RANK', -1)) != -1

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

def download_file_with_lock(url, filename, postprocess_fn=None):
    
    # only a single rank will download and call postprocess_fn

    base_dir = get_base_dir()
    file_path = os.path.join(base_dir, filename)
    lock_path = file_path + ".lock"

    if os.path.exists(file_path):
        return file_path
    
    with FileLock(lock_path):
        # only a single rank can acquire this lock, all others block until released

        if os.path.exists(file_path):
            return file_path

        print(f"downloading {url}...")
        with urllib.request.urlopen(url) as response:
            content = response.read()

        with open(file_path, 'wb') as f:
            f.write(content)
        print(f"downloaded to {file_path}")

        if postprocess_fn is not None:
            postprocess_fn(file_path)

    return file_path

class DummyWandb:
    def __init__(self):
        pass
    def log(self, *args, **kwargs):
        pass
    def finish(self):
        pass


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

