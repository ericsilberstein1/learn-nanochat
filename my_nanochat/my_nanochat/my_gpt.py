# my limited copy of gpt.py. See challenge-10-understand-model-architecture/understand-model-architecture.ipynb

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from my_nanochat.my_common import get_dist_info, log_memory_stats
from my_nanochat.muon import Muon, DistMuon

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

# NOT USED, HERE ONLY FOR EXPLORTATION IN CHALLENGE 15
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
		is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
	L, S = query.size(-2), key.size(-2)
	scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
	attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
	if is_causal:
		assert attn_mask is None
		temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
		attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

	if attn_mask is not None:
		if attn_mask.dtype == torch.bool:
			attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
		else:
			attn_bias = attn_mask + attn_bias

	if enable_gqa:
		key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
		value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

	log_memory_stats("before SDPA query @ ...", {}, 3)
	attn_weight = query @ key.transpose(-2, -1) * scale_factor
	log_memory_stats("after SDPA query @ ...", {'attn_weight': attn_weight}, 3)
	attn_weight += attn_bias
	attn_weight = torch.softmax(attn_weight, dim=-1)
	attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
	log_memory_stats("before SDPA computing attn_weight @ value", {}, 3)
	result = attn_weight @ value
	log_memory_stats("after SDPA computing attn_weight @ value", {'result': result}, 3)
	return result

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
        assert kv_cache is None # add support for this later
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        log_memory_stats("after ATTN c_q(x)", {'q': q}, 2)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        log_memory_stats("after ATTN c_k(x)", {'k': k}, 2)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        log_memory_stats("after ATTN c_v(x)", {'v': v}, 2)

        cos, sin = cos_sin
        q = apply_rottary_embed(q, cos, sin)
        log_memory_stats("after ATTN apply_rottary_embed(q, cos, sin)", {'resulting q': q}, 2)
        k = apply_rottary_embed(k, cos, sin)
        log_memory_stats("after ATTN apply_rottary_embed(k, cos, sin)", {'resulting k': k}, 2)
        q = norm(q)
        log_memory_stats("after ATTN norm(q)", {'resulting q': q}, 2)
        k = norm(k)
        log_memory_stats("after ATTN norm(k)", {'resulting k': k}, 2)

        q, k, v = q.transpose(2,1), k.transpose(2,1), v.transpose(2,1) # (B,T,H,D) -> (B,H,T,D)

        # code related to KV cache goes here

        # will understand and add code for GQA later
        assert self.n_head == self.n_kv_head
        enable_gqa = self.n_head != self.n_kv_head # always false for now

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
        log_memory_stats("after ATTN scaled_dot_product_attention", {'y': y}, 2)
        y = y.transpose(1,2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        log_memory_stats("after ATTN c_proj(y)", {'resulting y': y}, 2)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
    
    def forward(self, x):
        log_memory_stats("start of MLP forward", {'input x': x}, 2)
        x = self.c_fc(x)
        log_memory_stats("after MLP c_fc", {'resulting x': x}, 2)
        x = F.relu(x).square()
        log_memory_stats("after MLP F.relu(x).square()", {'resulting x': x}, 2)
        x = self.c_proj(x)
        log_memory_stats("after MLP c_proj", {'resulting x': x}, 2)
        return x

class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache):
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        log_memory_stats("after BLOCK x + attn(norm(x))", {'resulting x': x}, 1)
        x = x + self.mlp(norm(x))
        log_memory_stats("after BLOCK x + mlp(norm(x))", {'resulting x': x}, 1)
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
        DistAdamW = None # for now so it will fail until I "copy" adamw.py
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        
        # Create the Muon optimizer for the linear layers
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMoon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)

        # combine the two optimizers into one list
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"] # guessing for reporting

        return optimizers


    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        log_memory_stats("start of GPT forward", {'input idx': idx})
        assert kv_cache is None # for now
        
        B, T = idx.size()

        assert T < self.cos.size(1)
        assert idx.device == self.cos.device
        assert self.cos.dtype == torch.bfloat16

        T0 = 0 # TODO T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        x = self.transformer.wte(idx)
        log_memory_stats("after wte", {'resulting x': x})
        x = norm(x)
        log_memory_stats("after initial norm", {'resulting x': x})
        for i, block in enumerate(self.transformer.h):
            x = block(x, cos_sin, kv_cache)
            log_memory_stats(f"after block {i}", {'resulting x': x})
        x = norm(x)
        log_memory_stats("after final norm", {'resulting x': x})

        logits = self.lm_head(x)
        log_memory_stats("after lm_head", {'logits': logits})
        softcap = 15
        logits = softcap * torch.tanh(logits / softcap)
        log_memory_stats("after softcap * tanh(logits / softcap)", {'logits': logits})
        if targets is not None:
            logits = logits.float()
            log_memory_stats("after logits.float()", {'logits': logits})
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            log_memory_stats("after computing loss")
            return loss
        else:
            return logits


