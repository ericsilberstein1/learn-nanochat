import torch

class KVCache:

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers):
        # K and V each shape (B, H, T, D) (I was calling D C for channels in my notebook)
        self.kv_shape = (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
        self.kv_cache = None
        self.pos = 0

    def reset(self):
        self.pos = 0

    def get_pos(self):
        return self.pos
    
    def prefill(self, other):
        assert self.kv_cache is None, "cannot prefill non-empty KV cache"
        assert other.kv_cache is not None, "cannot prefill with a None KV cache"
        for ix, (dim1, dim2) in enumerate(zip(self.kv_shape, other.kv_shape)):
            if ix in [0, 1, 3, 5]:
                # num_layers, k/v, num_heads, head_dim (believe his comment is wrong)
                assert dim1 == dim2, f"Dim {ix} mismatch: {dim1} != {dim2}"
            elif ix == 2:
                assert dim1 == dim2 or dim2 == 1, f"Batch dim mismatch: {dim1} != {dim2}"
            elif ix == 4:
                # seq_len
                assert dim1 >= dim2, f"Seq len mismatch: {dim1} < {dim2}"
        
        # init
        dtype, device = other.kv_cache.dtype, other.kv_cache.device
        self.kv_cache = torch.empty(self.kv_shape, dtype=dtype, device=device)

        # copy data
        self.kv_cache[:, :, :, :, :other.pos, :] = other.kv_cache

        # update pos
        self.pos = other.pos

    def insert_kv(self, layer_idx, k, v):
        if self.kv_cache is None:
            self.kv_cache = torch.empty(self.kv_shape, dtype=k.dtype, device=k.device)
        B, H, T_add, D = k.size()
        t0, t1 = self.pos, self.pos + T_add
        # grow if needed
        if t1 > self.kv_cache.size(4):
            t_needed = t1 + 1024 # 1024 extra
            t_needed = (t_needed + 1023) & ~1023 # round up to nearest multiple of 1024
            additional_shape = list(self.kv_cache.shape)
            additional_shape[4] = t_needed - self.kv_cache.size[4]
            additional_cache = torch.empty(additional_shape, dtype=k.dtype, device=k.device)
            self.kv_cache = torch.cat([self.kv_cache, additional_cache], dim=4).contiguous()
            self.kv_shape = self.kv_cache.shape
        # insert into cache
        self.kv_cache[layer_idx, 0, :, :, t0:t1] = k
        self.kv_cache[layer_idx, 1, :, :, t0:t1] = v
        # return full keys/values up to pos
        key_view = self.kv_cache[layer_idx, 0, :, :, :t1]
        value_view = self.kv_cache[layer_idx, 1, :, :, :t1]
        # Increment pos after the last layer of the Transformer processes -- my comment: does that seem fragile?
        if layer_idx == self.kv_cache.size(0) - 1:
            self.pos = t1
        return key_view, value_view
