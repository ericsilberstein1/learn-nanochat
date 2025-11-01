
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

See note at top of `tokenize.ipynb` for how to build the rust part. (Don't use `cargo build`.)

#### Challenge 8
Download some of the actual text used to train the tokenizer and model, look at it, and then train the tokenizer from challenge 7 on it.

#### Challenge 9
Understand shape / contents of x and y in `x, y = next(trainLoader)`

Why? There is no way I can just start "copying" all the pre-training and model code without understanding it better first. After skimming `gpt.py`, `base_train.py`, and `dataloader.py`, I'm thinking it's better to understand the model first and then get to training. A potential starting point is to recreate enough of the model code to make sure I understand every line of `forward()`. However, a baby step even before that is to recreate enough of the dataloader so I understand exactly what x and y are in `x, y = next(train_loader)`. I assume x will be ~ a tensor with sequence length tokens by batch size starting with `<bos>` and y will be the same thing starting one token over and ending one token later.

#### Challenge 10
Create the model skeleton in a notebook and go through each step of the forward pass. Do this with tiny dimensions and a tiny x and y so I can see and follow the intermediate tensors.

#### Challenge 11
Understand the weight initialization code and copy it to `my_gpt.py`. Goal is to be able to pretrain the model soon.

#### Challenge 12
Understand the setup optimizers code and copy it to `my_gpt.py`. Still moving toward having enough in place to pretrain the model.

#### Challenge 13
Leaving out nearly as much as possible from `base_train.py`, try to train a depth 4 (?) model on my macbook.

#### Challenge 14
Train the baby model from challenge 13 on a single GPU. The point of this is to get an environment set up to work with a GPU and to make sure device is set/used properly in the code so far. I'll still need to add in validation, metrics, and checkpointing before scaling up.

#### Future potential challenges and/or things to look up and/or todo
* Evaluate the tokenizer similar to `scripts/tok_eval.py`.

* Understand the KV cache including how much compute is saved by it

* Why do we apply rotary embeddings to q and k but not v? What if we applied it only to one? Or all three?

* Why is bias=False for the linear transforms in GPT?

* Why is the type of cos / sin bfloat16?

* Make sure I understand why we init many layers to zero

* Replace my wholesale copy of muon.py with one I hand copied and understand

* Understand meta init

* With AdamW, for example, why do we still modify the LR in the training loop?

* How do positional embeddings really work? Why do they work? What if you leave them out? What if you put them only before the first layer?

* After challenge 14, add autodetect_device_type() so the same notebooks will work on cpu and gpu going forward
