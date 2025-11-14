import argparse
import os
import torch

from my_nanochat.my_tokenizer import MyTokenizer
from my_nanochat.my_dataset import text_iterator
from my_nanochat.my_common import get_base_dir

parser = argparse.ArgumentParser(description='Train a BPE tokenizer')
parser.add_argument('--max_chars', type=int, default=10_000_000_000, help='Maximum characters to train on (default: 10B)')
parser.add_argument('--doc_cap', type=int, default=10_000, help='Maximum characters per document (default: 10,000)')
parser.add_argument('--vocab_size', type=int, default=65536, help='Vocabulary size (default: 65536 = 2^16)')
args = parser.parse_args()
print(f"max_chars: {args.max_chars:,}")
print(f"doc_cap: {args.doc_cap:,}")
print(f"vocab_size: {args.vocab_size:,}")

print("starting to train tokenizer")
tokenizer = MyTokenizer.train_from_iterator(
    text_iterator(max_chars=args.max_chars, doc_cap=args.doc_cap),
    vocab_size=args.vocab_size)
print("finished training tokenizer")

tokenizer.save(os.path.join(get_base_dir(), "my-tokenizer.pkl"))

# Quick inline sanity check
test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text

# create and save token id -> number of bytes map
vocab_size = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())
token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
token_bytes = []
for token_id in range(vocab_size):
    token_str = token_strings[token_id]
    if token_str in special_set:
        token_bytes.append(0)
    else:
        id_bytes = len(token_str.encode("utf-8"))
        token_bytes.append(id_bytes)
token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device="cpu")
token_bytes_path = os.path.join(get_base_dir(), "token_bytes.pt")
torch.save(token_bytes, token_bytes_path)
print(f"Saved token_bytes to {token_bytes_path}")

# TODO add to report

