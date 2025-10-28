# my copy of dataloader.py. See challenge-09-understand-model-input/understand-model-input.ipynb

from collections import deque
import torch
from my_nanochat.my_tokenizer import get_tokenizer
from my_nanochat.my_dataset import parquets_iter_batched
from my_nanochat.my_common import get_base_dir, get_dist_info

def tokenizing_distributed_data_loader(B, T, split, tokenizer_threads=4, tokenizer_batch_size=128, device="cuda"):
    assert split in ["train", "val"]
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    needed_tokens = B * T + 1
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    token_buffer = deque()

    def document_batches():
        while True:
            for batch in parquets_iter_batched(split=split, start=ddp_rank, step=ddp_world_size):
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size]
    batches = document_batches()

    batch_index = 0
    while True:
        while len(token_buffer) < needed_tokens:
            doc_batch = next(batches)
            token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
            for tokens in token_lists:
                token_buffer.extend(tokens)
            batch_index += 1
        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
        # Why pin memory? ChatGPT: When training on a GPU, tensors must be copied from CPU memory to GPU memory.
        # Normally, this involves paging operations that can be relatively slow. But if the source tensor is in
        # pinned (page-locked) memory, the GPU can perform asynchronous, faster transfers using DMA
        # (Direct Memory Access).
        scratch = torch.tensor(tokens, dtype=torch.int64, pin_memory=(device == "cuda"))
        inputs_cpu = scratch[:-1].to(dtype=torch.int32)
        targets_cpu = scratch[1:]
        # non_blocking means: “If possible, don’t wait for this memory copy to finish before moving on — 
        # let it happen asynchronously.”
        inputs = inputs_cpu.view(B, T).to(device=device, dtype=torch.int32, non_blocking=True)
        targets = targets_cpu.view(B, T).to(device=device, dtype=torch.int64, non_blocking=True)
        yield inputs, targets

