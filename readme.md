
2025-Oct-20

I heard about [nanochat](https://github.com/karpathy/nanochat) and watched this [podcast](https://www.youtube.com/watch?v=lXUZvyajciY) with Andrej Karpathy. He suggests learning by rewriting all the code without any copy and paste.

I started to look at the tokenizer. I found [this](https://huggingface.co/learn/llm-course/en/chapter6/5) intro to byte-pair encoding tokenization on Hugging Face and read enough to get the idea.

#### Challenge 1
Write a toy BPE tokenizer in python to get the idea. Doing this in `challenge-01-play-tokenzier/tokenizer.ipynb`.

```
uv run jupyter lab
```

#### Challenge 2
I've never used Rust before. Get familiar with it by before trying to reimplement a performant tokenzier similart to the one in the nanogpt repo. Start with hello world.

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cargo new challenge-02-hello-world-rust
cd challenge-02-hello-world-rust
cargo run
```

#### Challenge 3
Rust version of the python toy code from tokenzier.ipynb.

```
cargo run
```

Got this working and then made the code into what is hopefully more idiomatic Rust.


#### Challlenge 4
Write something that does something in parallel, example using map / reduce.

#### Challenge 5
Call Rust code from Python.

See note at top of `call-rust-functions.ipynb` for how to build. (Don't use `cargo build`.)

#### Challenge 6
Now looking a bit more carefully at [scripts/tok_train.py](https://github.com/karpathy/nanochat/blob/master/scripts/tok_train.py) and [nanochat/tokenizer.py](https://github.com/karpathy/nanochat/blob/master/nanochat/tokenizer.py) in the nanochat repo I see that there are two tokenizer implementatins. One uses a combination of his [rustbpe](https://github.com/karpathy/nanochat/tree/master/rustbpe) and [tiktoken from openai](https://github.com/openai/tiktoken) and the other HuggingFace Tokenizer. There's also a lot more going on than in my toy versions.

Create a tiktoken object using hardcoded (not learned) tokens and use it to encode and decode text.

#### Challenge 7
Make a simplified version of tokenizer.py and rustbpe. I should be able to have a notebook that instantiates a tokenizer, gives it some text to train on, it in turn calls a rust library to do the actual training, and once trained, I can encode and decode. It should also be possible to save and load the tokenizer to disk.

This is similar to challenge 1 but it should be byte-level and scalable so I can then train on large amounts of text.

