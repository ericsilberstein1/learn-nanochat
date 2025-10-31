import rust_tokenizer;
import tiktoken;
import pickle;
from functools import lru_cache;
from my_nanochat.my_common import get_base_dir
import os

# copied from https://github.com/karpathy/nanochat/blob/master/nanochat/tokenizer.py
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class MyTokenizer:

    def __init__(self, enc: tiktoken.Encoding):
        self.enc = enc

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        tokenizer = rust_tokenizer.Tokenizer()
        tokenizer.train_from_iterator(text_iterator, vocab_size, buffer_size=8192, pattern=SPLIT_PATTERN);
        mergeable_ranks = tokenizer.get_mergeable_ranks()
        return cls(
            enc = tiktoken.Encoding(
                name = "my-encoding",
                pat_str = SPLIT_PATTERN,
                mergeable_ranks = dict(mergeable_ranks),
                special_tokens = {'<bos>': len(mergeable_ranks)}
            )
        )

    def encode(self, text, prepend=None, num_threads=8):

        if prepend is not None:
            assert(isinstance(prepend, int)) # for now at least, can enhance later to accept string or int

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend)
            return ids
        elif isinstance(text, list):
            batch = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for ids in batch:
                    ids.insert(0, prepend)
            return batch
        else:
            raise ValueError(f"invalid inpuyt type: {type(text)}")

    def decode(self, tokens):
        return self.enc.decode(tokens);

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.enc, f)
        print(f"Saved tokenizer to {filename}")

    @classmethod
    def load_from_file(cls, filename):
        with open(filename, 'rb') as f:
            enc = pickle.load(f)
        return cls(enc)

    @lru_cache(maxsize=32) # will take his word that encode_single_token is "slow"
    def encode_special(self, text):
        return self.enc.encode_single_token(text)
    
    def get_bos_token_id(self):
        return self.encode_special('<bos>') # TODO he decided it was worth it to hold onto it, maybe change to that?

    def get_vocab_size(self):
        return self.enc.n_vocab

def get_tokenizer():
    return MyTokenizer.load_from_file(os.path.join(get_base_dir(), 'my-tokenizer.pkl'))
