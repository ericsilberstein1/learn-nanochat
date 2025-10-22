## Log

### 2025-Oct-20

I heard about [NanoGPT](https://github.com/karpathy/nanoGPT) and watched this [podcast](https://www.youtube.com/watch?v=lXUZvyajciY) with Andrej Karpathy. He suggests learning by rewriting all the code without any copy and paste.

I started to look at the tokenizer. I found [this](https://huggingface.co/learn/llm-course/en/chapter6/5) intro to byte-pair encoding tokenization on Hugging Face and read enough to get the idea.

#### Challenge #1
Write a toy BPE tokenizer in python to get the idea. Doing that in tokenizer.ipynb.

```
uv run jupyter lab
```

### 2025-Oct-21

#### Challenge #2
I've never used Rust before. Before trying to reimplement the performant tokenzier in the repo, first reimplment the toy one from above in Rust.

#### Challenge 2.1
Hello world in Rust.

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cargo new hello_world_rust
cd hello_world_rust
cargo run
```

#### Challenge 2.2
Rust version of the python toy code from tokenzier.ipynb 

```
cd toy_tokenizer_rust
cargo run
```

#### Challlenge 2.3
Write something that runs in parallel, example using map / reduce.

#### Challenge 2.4
Call Rust code from Python.
