
import argparse
import torch
import sys
sys.path.append('../my_nanochat')
from my_nanochat.my_gpt import GPTConfig, GPT
from my_nanochat.my_common import autodetect_device_type, log_memory_stats

def create_model_and_do_one_training_step(depth,max_seq_len, device_batch_size):

    log_memory_stats("starting")
    
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
    log_memory_stats("after creating model")
    model.init_weights()
    log_memory_stats("after init_weights()")
    optimizers = model.setup_optimizers()
    log_memory_stats("after setup_optimizers()")
    x = torch.randint(0, vocab_size, (device_batch_size, max_seq_len), dtype=torch.int32, device=device)
    y = torch.randint(0, vocab_size, (device_batch_size, max_seq_len), dtype=torch.int64, device=device)
    log_memory_stats("after creating x and y()", {'x' : x, 'y' : y})

    for step in range(2):
        print(f"\n===== STARTING STEP {step} =====")
        with autocast_ctx:
            loss = model(x,y)
            log_memory_stats("after forward")
        loss.backward()
        log_memory_stats("after backward")
        for i, opt in enumerate(optimizers):
            opt.step()
            log_memory_stats(f"After optimizer {i} step")

    log_memory_stats("ending")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--max-seq-len", type=int, required=True)
    parser.add_argument("--device-batch-size", type=int, required=True)
    args = parser.parse_args()

    create_model_and_do_one_training_step(args.depth, args.max_seq_len, args.device_batch_size)
