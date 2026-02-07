import torch
import torch.nn.functional as F
import tiktoken
from train_gpt2 import GPT,GPTConfig
import math
import numpy as np


# class DataLoaderLite:
#     def __init__(self, B, T, split, val_fraction=0.1):
#         self.B = B
#         self.T = T
#         bin_file = f"{split}.bin"

#         # Just memory-map the bin file, no text read
#         self.tokens = np.memmap(bin_file, dtype=np.uint16, mode="r")
#         self.num_tokens = len(self.tokens)
#         self.current_position = 0

#         self.grad_accum_steps = None
#         self.steps_needed = None

#         print(f"{split} split loaded with {self.num_tokens} tokens "
#               f"({self.num_tokens // (B*T)} batches, streaming mode)")

#     def set_grad_accum(self, grad_accum_steps):
#         self.grad_accum_steps = grad_accum_steps
#         tokens_per_step = self.B * self.T * self.grad_accum_steps
#         self.steps_needed = math.ceil(self.num_tokens / tokens_per_step)
#         print(f"With grad_accum_steps={grad_accum_steps}, steps_needed={self.steps_needed}")

#     def reset(self):
#         self.current_position = 0

#     def next_batch(self):
#         B, T = self.B, self.T
#         start = self.current_position
#         end = start + (B * T + 1)

#         if end > self.num_tokens:
#             self.reset()
#             start = 0
#             end = B * T + 1

#         buf = torch.from_numpy(self.tokens[start:end].astype(np.int64))
#         x = buf[:-1].view(B, T).contiguous()
#         y = buf[1:].view(B, T).contiguous()

#         self.current_position = start + (B * T)
#         return x, y

# Recreate the model with the same config
model = GPT(GPTConfig(vocab_size=50257))
checkpoint = torch.load("gpt_checkpoint.pth", map_location='cpu')
model.load_state_dict(checkpoint['model'])  # or 'cuda' if using GPU
model.eval()


num_return_sequence = 5
max_length = 90
context_window = 60

enc = tiktoken.get_encoding('gpt2')
prompt = "Once upon a time, a little girl goes to play,"
tokens = torch.tensor(enc.encode(prompt), dtype=torch.long).unsqueeze(0)  # (1, seq_len)
tokens = tokens.cpu()
model.cpu()
model.eval()
print(model.state_dict())
generated_sequences = []

for seq_idx in range(num_return_sequence):
    tokens_gen = tokens.clone()
    with torch.no_grad():
        for _ in range(max_length):
            input_tokens = tokens_gen[:, -context_window:]
            logits, _ = model(input_tokens)
            next_token_logits = logits[:, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens_gen = torch.cat([tokens_gen, next_token], dim=1)
    generated_sequences.append(tokens_gen)

with open("generated.txt", "w", encoding="utf-8") as f:
    for i, seq in enumerate(generated_sequences):
        text = enc.decode(seq[0].tolist())
        f.write(f"Generated {i+1}: {text}\n\n")