
import argparse
import torch
import sys
sys.path.append('../my_nanochat')
from my_nanochat.my_gpt import GPTConfig, GPT
from my_nanochat.my_common import autodetect_device_type, memory_stats

def create_model_and_do_one_training_step(depth,max_seq_len, device_batch_size):

    print(f"Before: {memory_stats()}")
    
    device = autodetect_device_type()

    num_layers = depth
    model_dim = depth * 64
    num_heads = max(1, (model_dim + 127) // 128)
    num_kv_heads = num_heads
    vocab_size = 65537
    total_batch_size = device_batch_size * max_seq_len

    model_config_kwargs = dict(
        sequence_len=max_seq_len,
        vocab_size=vocab_size,
        n_layer=num_layers,
        n_head=num_heads,
        n_kv_head=num_kv_heads,
        n_embd=model_dim
    )
    print(model_config_kwargs)

    autocast_ctx = torch.amp.autocast(device_type=device, dtype=torch.bfloat16) if device == "cuda" else nullcontext()
    with torch.device("meta"):
        model_config = GPTConfig(**model_config_kwargs)
        model = GPT(model_config)
    model.to_empty(device=device)
    print(f"After creating model: {memory_stats()}")
    model.init_weights()
    print(f"After initializing weights: {memory_stats()}")
    optimizers = model.setup_optimizers()
    print(f"After setting up optimizers: {memory_stats()}")
    x = torch.randint(0, vocab_size, (device_batch_size, max_seq_len), dtype=torch.int32, device=device)
    y = torch.randint(0, vocab_size, (device_batch_size, max_seq_len), dtype=torch.int64, device=device)
    print(f"After creating x and y: {memory_stats()}")
    with autocast_ctx:
        loss = model(x,y)
        print(f"After forward: {memory_stats()}")
    loss.backward()
    print(f"After backward: {memory_stats()}")
    for i, opt in enumerate(optimizers):
        opt.step()
        print(f"After optimizer {i} step: {memory_stats()}")

    print(f"After: {memory_stats()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--max-seq-len", type=int, required=True)
    parser.add_argument("--device-batch-size", type=int, required=True)
    args = parser.parse_args()

    create_model_and_do_one_training_step(args.depth, args.max_seq_len, args.device_batch_size)
