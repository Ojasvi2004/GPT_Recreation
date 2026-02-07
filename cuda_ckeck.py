import torch

print("CUDA Available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_capability(0))

# print("BF16 matmul available:", torch.cuda.is_bf16_supported())
# print("BF16 AMP available:", torch.cuda.amp.common._is_bf16_available())
