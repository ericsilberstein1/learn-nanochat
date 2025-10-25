import rust_tokenizer;
import tiktoken;
import pickle;

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

    def encode(self, text):
        return self.enc.encode_ordinary(text)

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
