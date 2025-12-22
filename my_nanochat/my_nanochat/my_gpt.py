# my limited copy of gpt.py. See challenge-10-understand-model-architecture/understand-model-architecture.ipynb

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from my_nanochat.my_common import get_dist_info
from my_nanochat.muon import Muon, DistMuon
from my_nanochat.adamw import DistAdamW

@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

def apply_rottary_embed(x, cos, sin):
    assert x.ndim == 4  # (B, T, H, n_embed / n_head)
    d = x.shape[3] // 2
    x1, x2 = x[:,:,:,:d], x[:,:,:,d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat((y1, y2), 3)
    out = out.to(x.dtype)
    return out

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head
        assert self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache=None):
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        cos, sin = cos_sin
        q = apply_rottary_embed(q, cos, sin)
        k = apply_rottary_embed(k, cos, sin)
        q = norm(q)
        k = norm(k)

        q, k, v = q.transpose(2,1), k.transpose(2,1), v.transpose(2,1) # (B,T,H,D) -> (B,H,T,D)

        # apply KV cache: insert current k,v into cache and get the full view so far
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2) # number of queries in this forward pass (I think will usually be 1)
        Tk = k.size(2) # number of keys/values in total (in the cache + in this forward pass)

        # code related to KV cache goes here

        # will understand and add code for GQA later
        assert self.n_head == self.n_kv_head
        enable_gqa = self.n_head != self.n_kv_head # always false for now

        # read notes in challenge-20-understand-engine/add-kv-cache-support-to-gpt.ipynb
        if kv_cache is None or Tq == Tk:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
        elif Tq == 1:
            # believe this is the common case during inference after the initial prompt is processed
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
        else:
            attn_mask = torch.zeros((Tq,Tk), dtype=torch.bool, device=q.device) # True = keep
            prefix_len = Tk - Tq # 2
            if prefix_len > 0: # he says can't be negative but could be zero but don't think can be 0 due to above
                attn_mask[:, :prefix_len] = True    # so whatever is in the prefix is allowed
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            # A "square" of Trues on the left and a "triangle" of Trues on the right, like
            # if Tk = 4 and Tq = 2 we'll end up with
            # True True True False
            # True True True True
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)

        y = y.transpose(1,2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache):
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # see note in his code about how this is a "fake" init of rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10_000, device=None):
        if device is None:
            device = self.get_device()
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]  # B, T, H, C  ??
        return cos, sin

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)  # number of output features
            fan_in = module.weight.size(1)   # number of input features
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init_zeroes_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    def get_device(self):
        return self.transformer.wte.weight.device

    def init_weights(self):
        self.apply(self._init_weights)
        # zero out the classifier weights
        torch.nn.init.zeros_(self.lm_head.weight);
        # zero out c_proj weights in all blocks
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        l, h, q, t = self.config.n_layer, self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token
    
    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.2, weight_decay=0.0):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        # seperate params into 3 groups
        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params)

        # create the AdamW optimizer for the embedding and lm_head
        # see notes in his code and notebook
        dmodel_lr_scale = (model_dim / 768) ** -0.5; dmodel_lr_scale
        if rank == 0:
            print(f"Scaling the LR for the AdamW parameters proportional to 1/sqrt({model_dim}/768) = {dmodel_lr_scale}")
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        ]
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        
        # Create the Muon optimizer for the linear layers
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)

        # combine the two optimizers into one list
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"] # guessing for reporting

        return optimizers


    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        
        B, T = idx.size()

        assert T < self.cos.size(1)
        assert idx.device == self.cos.device
        assert self.cos.dtype == torch.bfloat16

        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        x = self.transformer.wte(idx)
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)
        x = norm(x)

        logits = self.lm_head(x)
        softcap = 15
        logits = softcap * torch.tanh(logits / softcap)
        if targets is not None:
            logits = logits.float()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            return logits

    @torch.inference_mode() # like no_grad() but with more restrictions and speedup
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # we just do bs=1
        for _ in range(max_tokens):
            logits = self.forward(ids) # [B, len(ids), vocab_size] ?
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf') # set all logits less than the smallest of the top k to -inf
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token















