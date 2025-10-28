import os

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