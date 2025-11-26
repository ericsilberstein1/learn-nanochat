
2025-Oct-20

I heard about [nanochat](https://github.com/karpathy/nanochat) and watched this [podcast](https://www.youtube.com/watch?v=lXUZvyajciY) with Andrej Karpathy. He suggests learning by rewriting all the code without any copy and paste.

Each challenge is in a folder. Most have jupyter notebooks.

```
uv sync
uv run jupyter lab
```

### Challenges

#### Challenge 1
Write a toy BPE tokenizer in python to get the idea. Doing this in `challenge-01-play-tokenzier/tokenizer.ipynb`.

#### Challenge 2
I've never used Rust before. Get familiar with it by before trying to reimplement a performant tokenzier similart to the one in the nanochat repo. Start with hello world.

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cargo new challenge-02-hello-world-rust
cd challenge-02-hello-world-rust
cargo run
```

#### Challenge 3
Code a Rust version of the python toy code from `challenge-01-play-tokenzier/tokenizer.ipynb`.

```
cargo run
```

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

#### Challenge 15
Sticking with GPU for a bit, understand how to calculate the GPU memory needed for the model.

#### Challenge 16
Do a tiny example of back propagation by hand. This is to build intuition for what additional memory gets allocated during back prop which came up in challenge 15.

#### Challenge 17
Create `scripts/my_base_train.py` by copying much of `base_train.py`. This should include the code from the baby training in previous challenges, code to pass config options on the command line, and code to save checkpoints. The idea is get organized before adding evaluations and scaling up training.

#### Challenge 18
Add calculating and displaying the BPB (bits per byte) evaluation to training.

#### Challenge 19
Train for hours on the GPU. Now that `base_train.py` and BPB evaluation are in place, try running a longer training (e.g. in tmux) and confirm it completes.

#### Challenge 20
Understand and add `engine.py`. So far, for example at the end of the challenge 19 notebook, I've been generating samples in a naive way. In his training loop I see he samples every so often and does so through this "engine." This may also be where I learn about the KV cache. (Yes, that's essential. Making a notebook just for that part in the challenge.)

#### Challenge 21
Understand CORE metric evaluation. Possibly implement now or possibly hold off. (Decided to implement.)

#### Challenge 22
Add wandb logging. This should be straightforward and maybe it will be helpful before scaling up training.

#### Challenge 23
Train on multiple GPUs. Before trying out more powerful GPUs, I want to get the hang of training with multiple GPUs.

#### Challenge 24
Train on 2 GPUs for a few hours.

#### Challenge 25
Do a complete pretrain of the d20 model using the same specs as in his `speedrun.sh`.

#### Challenge 26
Understand what "mid train" is and look at some of the training data.

#### Challenge 27
Understand chat eval.

#### Challenge 28
Midtrain the d20 model.

#### Challenge 29
Understand SFT.

#### Challenge 30
SFT train the d20 model.

#### Challenge 31
Add tool calling to engine.

#### Challenge 32
Redo chat eval on d20 and repeat SFT train with more frequent evaluation of validation loss.

#### Challenge 33
Add chat CLI.

#### Challenge 34
Understand and add reporting.

#### Challenge 35
Understand reinforcement learning.

#### Challenge 36
RL train the d20 model.

### Examples of training/validation/metrics data

* [For training tokenizer and base training of model](challenge-08-train-tokenizer/example-data.ipynb)
* [For CORE metrics](challenge-21-understand-core-metric/core-evaluation-data-examples.ipynb)
* [For mid-train](challenge-26-understand-midtrain/midtrain-data-examples.ipynb)
* [For chat eval](challenge-27-understand-chat-eval/chat-eval-data-examples.ipynb)
* [Additional for SFT training](challenge-29-understand-sft/sft-data-examples.ipynb)

### Future potential challenges, things to look up, questions, and todo
* Evaluate the tokenizer similar to `scripts/tok_eval.py`.

* Understand the KV cache including how much compute is saved by it (first part done, see challenge 20)

* Why do we apply rotary embeddings to q and k but not v? What if we applied it only to one? Or all three?

* Why is bias=False for the linear transforms in GPT?

* Why is the type of cos / sin bfloat16?

* Make sure I understand why we init many layers to zero

* Replace my wholesale copy of muon.py with one I hand copied and understand

* Understand meta init

* With AdamW, for example, why do we still need to modify the LR in the training loop?

* How do positional embeddings really work? Why do they work? What if you leave them out? What if you put them only before the first layer?

* Understand how to compute possible model dimensions, batch size, sequence length, etc. based on GPU memory. (mostly done, see challenge 15)

* For training at least, what is the difference between say a batch size of 1 and a sequence length of 64 vs a batch size of 2 and a sequence length of 32? How do you choose?

* Understand this: UserWarning: Quadro RTX 4000 does not support bfloat16 compilation natively, skipping (for example, if a GPU doesn't support bfloat16 compilation, is it better not to use bfloat16 at all?)

* Why in the dataloader do we use int32 for inputs but int64 for targets?

* Better understand the concept behind BPB eval, nits, why it's a legit way to compare loss across diff vocab sizes, etc.

* Make sure I understand how weight initialization works for distributed training and if it's ok for weights on diffrent GPUs to start out different (most done, see challenge 23)

* Try tiny examples of torch.dist functions in a notebook and potentially time moving tensors between GPUs vs within a GPU

* If warnings at start of train persist on more modern GPUs investigate

* Understand why we don't do grad clipping in midtraining, or why we need to do it in base training

* Understand why we don't evalute CORE metrics or sample every so often during mid training

* Re-implement or at least look more closely at execution.py and humaneval.py
