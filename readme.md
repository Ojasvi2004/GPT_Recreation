# NanoGPT From Scratch (PyTorch)

A lightweight, from-scratch implementation of a GPT-style Transformer language model built in **PyTorch**.
This project includes:

* A custom GPT architecture
* A streaming dataloader for large token datasets
* Mixed-precision training with gradient accumulation
* Checkpoint saving/loading
* Text generation (inference) after training

The goal is to understand how modern LLMs like GPT-2 work internally by implementing every major component manually.

---

## 🚀 Features

* Custom GPT architecture (decoder-only transformer)
* Multi-head causal self-attention using fused PyTorch kernels
* Weight tying between embedding and output layer
* Memory-efficient streaming dataloader using `numpy.memmap`
* Gradient accumulation for large effective batch sizes
* Cosine learning rate scheduler with warmup
* Automatic checkpoint resume
* Mixed precision training (bfloat16 autocast)
* Text generation with sampling

---

## 🧠 Model Architecture

The model is a simplified GPT-style transformer composed of:

* **Token embeddings**
* **Positional embeddings**
* Stacked transformer decoder blocks:

  * LayerNorm
  * Causal Self-Attention
  * Feed-forward MLP
* Final LayerNorm
* Language modeling head (tied weights)

Default config:

```python
GPTConfig(
    block_size=1024,
    vocab_size=50257,
    n_layer=6,
    n_head=4,
    n_embd=256
)
```

This configuration is intentionally small so it can train on consumer GPUs.

---

## 📂 Project Structure

```
.
├── train_gpt2.py        # Model + training script
├── train.bin            # Training token dataset
├── val.bin              # Validation token dataset
├── gpt_checkpoint.pth   # Saved checkpoint
├── generated.txt        # Generated text output
```

---

## 📦 Requirements

* Python 3.10+
* PyTorch (CUDA recommended)
* NumPy
* transformers
* tiktoken

Install dependencies:

```bash
pip install torch numpy transformers tiktoken
```

---

## 🗃 Dataset Preparation

The dataloader expects tokenized data stored as `.bin` files containing GPT-2 token IDs (`uint16`).

You must:

1. Tokenize your text dataset using `tiktoken`
2. Save tokens as:

```python
tokens.astype(np.uint16).tofile("train.bin")
tokens.astype(np.uint16).tofile("val.bin")
```

The dataloader streams directly from disk using memory mapping, so large datasets are supported.

---

## 🏋️ Training

Run training:

```bash
python train_gpt2.py
```

Training includes:

* Gradient accumulation
* Mixed precision training
* Cosine LR decay with warmup
* Periodic validation
* Automatic checkpoint saving

If a checkpoint exists, training resumes automatically.

---

## 💾 Checkpoints

The model saves:

```
gpt_checkpoint.pth
```

It contains:

* Model weights
* Optimizer state

This allows seamless resume of training.

---

## ✨ Inference (Text Generation)

After training, the script generates text from a prompt:

```python
prompt = "Once upon a time,"
```

The model samples tokens autoregressively and writes output to:

```
generated.txt
```

Multiple sequences are generated using probabilistic sampling.

---

## ⚙️ Optimization Techniques Used

* Scaled dot-product attention (kernel fusion)
* Gradient clipping
* Mixed precision (bfloat16 autocast)
* Weight decay parameter grouping
* Weight tying
* Streaming dataloader

These techniques improve speed and memory efficiency.

---

## 🎯 Learning Goals

This project is designed to:

* Teach transformer internals
* Show how GPT-style models are built from scratch
* Demonstrate efficient training loops
* Provide a foundation for experimenting with LLMs

It’s ideal for students and engineers learning deep learning systems.

---

## 🔮 Future Improvements

* Distributed training (DDP)
* Flash attention
* Larger model configs
* Tokenizer training
* Dataset pipeline automation
* Web demo interface

---

## 📜 License

MIT License

---

## 🙌 Acknowledgements

Inspired by GPT architecture and modern transformer research.
Built for educational purposes to explore large language models.
