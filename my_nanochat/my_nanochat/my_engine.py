import torch
import torch.nn.functional as F
from contextlib import contextmanager
import signal
import warnings
from collections import deque

@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}' timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)

def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', SyntaxWarning)
                return eval(formula, {"__builtins__": {}}, {})
    except Exception as e:
        signal.alarm(0)
        # see his comment about ok to ignore wrong calculator usage
        return None

def use_calculator(expr):
    # remove commas from numbers -- is that cheating?
    expr = expr.replace(",", "")

    if all([x in "0123456789*+-/.()" for x in expr]):
        if "**" in expr:
            return None
        return eval_with_timeout(expr)

    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
    if not all([x in allowed_chars for x in expr]):
        return None

    dangerous_patterns = ['__', 'import', 'exec', 'eval', 'compile', 'open', 'file',
                         'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
                         'getattr', 'setattr', 'delattr', 'hasattr']
    expr_lower = expr.lower()
    if any(pattern in expr_lower for pattern in dangerous_patterns):
        return None

    if '.count(' not in expr:
        return None

    return eval_with_timeout(expr)



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


class RowState:
    # Per-row state tracking during generation -- is a row like one stream of tokens that is getting built up? We process
    # rows through the model in in parallel, one column of tokens at a time?
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or [] # current token sequence for this row
        self.forced_tokens = deque()
        self.in_python_block = False
        self.python_expr_tokens = []
        self.completed = False # done generating this row, for example becuase hit <bos> ?

@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    # logits shape: (B, vocab_size)
    # return shape: (B, 1)
    assert temperature >= 0
    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)

class Engine:

    def __init__(self, model, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer # TODO to get BOS token for now, later for other tokens for tool use

    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        assert isinstance(tokens, list) and isinstance(tokens[0], int), 'expecting tokens to be a list of ints'
        device = self.model.get_device()
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        # TODO: get the special tokens we need to coordinate the tool use state machine
        python_start = self.tokenizer.encode_special("<|python_start|>")
        python_end = self.tokenizer.encode_special("<|python_end|>")
        output_start = self.tokenizer.encode_special("<|output_start|>")
        output_end = self.tokenizer.encode_special("<|output_end|>")
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>") # if sampled, ends row
        bos = self.tokenizer.get_bos_token_id()

        # 1) run a batch of size 1 with the prompt tokens to prefill the kv cache (?)
        m = self.model.config
        kv_model_kwargs = {
            'num_heads': m.n_kv_head,
            'head_dim': m.n_embd // m.n_head,
            'num_layers': m.n_layer,
        }
        kv_cache_prefill = KVCache(batch_size=1, seq_len=len(tokens), **kv_model_kwargs)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
        logits = logits[:, -1, :]
        next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)
        sampled_tokens = next_ids[:, 0].tolist()

        # 2) replicate the KV cache for each sample/row
        kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else self.model.config.sequence_len
        kv_cache_decode = KVCache(batch_size=num_samples, seq_len=kv_length_hint, **kv_model_kwargs)
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill

        # 3) initalize row states for each sample
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)] # because all sampeles start with same prompt (tokens)

        # 4) main generation loop
        num_generated = 0
        first_iteration = True
        while True:

            # stop conditions
            if max_tokens is not None and num_generated >= max_tokens:
                break
            if all(state.completed for state in row_states):
                break

            if first_iteration:
                sampled_tokens = [sampled_tokens[0]] * num_samples
                # funny my first thought on ^ is then we're forcing all samples to have the same "next" first token after
                # the prompt and he has a TODO for that
                first_iteration = False
            else:
                logits = self.model.forward(ids, kv_cache=kv_cache_decode) # (B, T, vocab_size) # CHECK ids
                assert ids.size(1) == 1 # TODO I ADDED THIS ASSERT because I'm curious when it could ever not be 1
                logits = logits[:, -1, :]
                next_ids = sample_next_token(logits, rng, temperature, top_k) # (B, 1)
                sampled_tokens = next_ids[:, 0].tolist()

            # Process each row
            token_column = [] # contains next token id along each row
            token_masks = [] # contains the mask 1 = it was sample, 0 = it was forced along each row
            for i, state in enumerate(row_states):
                is_forced = len(state.forced_tokens) > 0
                token_masks.append(0 if is_forced else 1)
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token) 

                # update state of row to include next token
                state.current_tokens.append(next_token)

                if next_token == assistant_end or next_token == bos:
                    state.completed = True

                if next_token == python_start:
                    state.in_python_block = True
                    state.python_expr_tokens = []
                elif next_token == python_end and state.in_python_block:
                    state.in_python_block = False
                    if state.python_expr_tokens:
                        expr = self.tokenizer.decode(state.python_expr_tokens)
                        result = use_calculator(expr)
                        if result is not None:
                            result_tokens = self.tokenizer.encode(str(result))
                            state.forced_tokens.append(output_start)
                            state.forced_tokens.extend(result_tokens)
                            state.forced_tokens.append(output_end)
                    state.python_expr_tokens = [] # do we need this and the one above?
                elif state.in_python_block:
                    state.python_expr_tokens.append(next_token)

            yield token_column, token_masks
            num_generated += 1

            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        assistant_end = self.tokenizer.encode_special('<|assistant_end|>')
        bos = self.tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            if all(completed):
                break
        return results, masks

