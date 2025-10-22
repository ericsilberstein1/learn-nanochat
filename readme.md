
2025-Oct-20

I heard about [NanoGPT](https://github.com/karpathy/nanoGPT) and watched this [podcast](https://www.youtube.com/watch?v=lXUZvyajciY) with Andrej Karpathy. He suggests learning by rewriting all the code without any copy and paste.

I started to look at the tokenizer. I found [this](https://huggingface.co/learn/llm-course/en/chapter6/5) intro to byte-pair encoding tokenization on Hugging Face and read enough to get the idea.

#### Challenge 1
Write a toy BPE tokenizer in python to get the idea. Doing this in `challenge-01-play-tokenzier/tokenizer.ipynb`.

```
uv run jupyter lab
```

#### Challenge 2
I've never used Rust before. Get familiar with it by before trying to reimplement a performant tokenzier similart to the one in the nanogpt repo.

#### Challenge 2.1
Hello world in Rust.

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cargo new challenge-02.1-hello-world-rust
cd challenge-02.1-hello-world-rust
cargo run
```

#### Challenge 2.2
Rust version of the python toy code from tokenzier.ipynb. Doing this in `challenge-02.2-play-tokenizer-rust`.

```
cargo run
```

Got this working and then made the code into what is hopefully more idiomatic Rust.

## Next challenges

#### Challlenge 2.3
Write something that runs in parallel, example using map / reduce.

#### Challenge 2.4
Call Rust code from Python.
