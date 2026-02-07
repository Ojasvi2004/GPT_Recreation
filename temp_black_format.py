from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math


@dataclass


class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 6
    n_head: int = 4
    n_embd: int = 256


config = GPTConfig(
    block_size=1024,   # keep or reduce to 256-512
    vocab_size=50257,  # fixed by tokenizer
    n_layer=6,         # fewer layers
    n_head=4,          # fewer attention heads
    n_embd=256         # smaller embedding dimension
)


    
class CasualSelfAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head==0
        #key,value,query projections concatenated in a single batch
        self.c_attn=nn.Linear(config.n_embd,3*config.n_embd)
        #output projetion matrix
        self.c_proj=nn.Linear(config.n_embd,config.n_embd)
        self.c_proj.NANO_GPT_SCALE_INIT= 1
   
        self.n_head=config.n_head
        self.n_embd=config.n_embd
        
        #casual mask to use in forward function for masked attention
        bias = torch.tril(torch.ones(config.block_size, config.block_size))
        bias = bias.view(1, 1, config.block_size, config.block_size)
        self.register_buffer("bias", bias)

        
    
    def forward(self,x):
        #batch size, sequence length, embedding dimensionality(n_embd)
        B,T,C=x.size()
        
        qkv=self.c_attn(x)
        q,k,v=qkv.split(self.n_embd,dim=2)
        
        #instead for diving the input matrices initially and then calculating Wv,Wk,Wq for each head we calculated it at once then then divided Q,K,V into the heads
        #saves the multiple matrix mutiplication, faster then conceptual way
        k=k.view(B,T,self.n_head,C // self.n_head).transpose(1,2)
        q=q.view(B,T,self.n_head,C // self.n_head).transpose(1,2)
        v=v.view(B,T,self.n_head,C // self.n_head).transpose(1,2)
        
        # #calculating attention score matrix      
        # att=(q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
        # #masking the attention matrix beacause gpt is a regressive model at inference time
        # att=att.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf'))   
        # #softmak converts attention score matrix into weights
        # att=F.softmax(att,dim=-1)
        
        # y=att@v
        # y=y.transpose(1,2).contiguous().view(B,T,C)
        
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True)   #faster because of kernel fusion
        
        y=y.transpose(1,2).contiguous().view(B,T,C)
        
        y=self.c_proj(y)
        return y
             
 


class MLP(nn.Module):
    # feed forward neural network
    def __init__(self,config):
        super().__init__()
        self.c_fc=nn.Linear(config.n_embd,4 * config.n_embd)
        self.gelu=nn.GELU(approximate='tanh')
        self.c_proj=nn.Linear(4*config.n_embd,config.n_embd)
        self.c_proj.NANO_GPT_SCALE_INIT= 1
        
    def forward(self,x):
        x=self.c_fc(x)
        x=self.gelu(x)
        x=self.c_proj(x)
        return x


class Block(nn.Module):
    #define the decoder block
    def __init__(self,config):
        super().__init__()
        self.ln_1=nn.LayerNorm(config.n_embd)
        self.attn=CasualSelfAttention(config)
        self.ln_2=nn.LayerNorm(config.n_embd)
        self.mlp=MLP(config)
    
    def forward(self,x):
        x=x + self.attn(self.ln_1(x))
        x=x + self.mlp(self.ln_2(x))
        return x

import inspect
     
class GPT(nn.Module):  #We use nn.Module so PyTorch can automatically manage parameters, submodules, training behavior.
    
    def __init__(self, config):
        super().__init__()     # Calls the parent constructors and sets up and track  Submodular,Paramter and Buffer registories for the upcoming custom layers
        self.config=config
        
        self.transformer=nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size,config.n_embd), #token embd
                wpe =nn.Embedding(config.block_size,config.n_embd), #pos encoding
                h=nn.ModuleList([Block(config)  for i in range(config.n_layer)]), # stack up layes for decoder blocks
                ln_f=nn.LayerNorm(config.n_embd),  #final layer normalization
            )
        )
        self.lm_head=nn.Linear(config.n_embd,config.vocab_size,bias=False)
        
        #weight sharing scheme
        self.transformer.wte.weight=self.lm_head.weight
        
    def __init_weights(self,module):
        std=0.02
        if hasattr(model,'NANO_GPT_SCALE_INIT'):
            std *= (2* self.config.n_layer)** -0.5
        if isinstance(module,nn.Linear):
            torch.nn.init.normal__(module.weight,mean=0.0,std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
            
        
    
    
    def forward(self,idx,targets=None):
        #idx of shape(B,T)
        B,T=idx.size()
        assert T<=self.config.block_size, f"Cannot forward a sequense of length ({B},{T})"
        
        pos=torch.arange(0,T,dtype=torch.long,device=idx.device)
        pos_emb=self.transformer.wpe(pos)  #Position enbedding of shape (T,n_embd)
        tok_emb=self.transformer.wte(idx)  #Token embedding of shape (B,T,N_embd)
        x=tok_emb+pos_emb
        
        #forward to the block of transformer
        
        for block in self.transformer.h:
            x=block(x)
        
        #forward to the final layernorm and the classifier
        x=self.transformer.ln_f(x)
        logits=self.lm_head(x) #(B,T,vocab_size)
        loss=None
        if targets is not None:
            loss=F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1))
        
        return logits,loss
        
    
    @classmethod   
    def from_pretrained(cls,model_type):
        """Loads Pretrained  GPT-2 Model Weights From Huggingface"""
        assert model_type in {'gpt2','gpt2-medium','gpt2-large','gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print('Loading weights from pretrained gpt2: %s' % model_type)
        
        #n_layesr,n_heads,n_embd are determined from model__type
        config_args={
            'gpt2':  dict(n_layer=6,n_head=4,n_embd=256), #124M params  changed to 30M for tinystory dataset
            'gpt2-medium': dict(n_layer=24,n_head=16,n_embd=1024),
            'gpt2-large':dict(n_layer=36,n_head=20,n_embd=1280),
            'gpt2-xl':dict(n_layer=48,n_head=25,n_embd=1600)#1558M params
        }[model_type]
        
        config_args['vocab_size']=50257 #constant for gpt model checkpoints
        config_args['block_size']=1024  #constant for gpt model checkpoints
        
      
        #initializes your model with given **configuration
        config=GPTConfig(**config_args)
        model=GPT(config)
        #take all the parametes from initialized model-(randomly initialized parameters)
        sd=model.state_dict()
        # print(sd.keys())
        sd_keys=sd.keys()
        sd_keys=[k for k in sd_keys if not k.endswith('.attn.bias')]#discard non learning buffer
        
        #init huging face transformer model
        model_hf=GPT2LMHeadModel.from_pretrained(model_type)
        #take all the parameters from hugging face models
        sd_hf=model_hf.state_dict()
        # print(sd_hf.keys())
        
        #copy params from huggingface model to my coustom model
        sd_keys_hf=sd_hf.keys()
        sd_keys_hf=[k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf=[k for k in sd_keys if not k.endswith('.attn.bias')]
        transposed=['attn.c_attn.weight','attn.c_proj.weight','mlp.c_fc.weight','mlp.c_proj.weight']
        
        assert len(sd_keys_hf)==len(sd_keys),f"mismatched keys :{len(sd_keys_hf)} 1= {len(sd_keys)}"
        
        for k in sd_keys_hf:
            if any (k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1]==sd[k].shape
                
                #in traing we use backpropagation so we needs the tracking of gradients so pytorch automatically do it
                # use torch.no_grad() to stop tracking the gradients by pytorch as this is inference not traning
             
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        
        return model
    
    def configure_optimizers(self,weight_decay,learning_rate,device):
        #start with all the candidate parameters
        param_dict={pn: p for pn, p in self.named_parameters()}
        param_dict={pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        #create optim groups. Any parameters that is in 2D will be weight decayed(like matmuls and embeddings) otherwise not(like bias and leayernorms)
        decay_params=[p for n, p in param_dict.items() if p.dim()>=2]
        nodecay_params=[p for n, p in param_dict.items() if p.dim()<2]
        
        optim_groups=[
            {'params':decay_params,'weight_decay':weight_decay},
            {'params':nodecay_params,'weight_decay':0.0}
        ]
        num_decay_params=sum(p.numel() for p in decay_params)
        num_nodecay_params=sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameters tensors:{len(decay_params)} with {num_decay_params} parameters")
        print(f"num nodecayed parameters tensors:{len(nodecay_params)} with {num_nodecay_params} parameters")
        fused_available='fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused=fused_available and 'cuda' in device
        print(f"using fused adam{use_fused}")
        optimizer=torch.optim.AdamW(optim_groups,lr=learning_rate,betas=(0.9,0.95),eps=1e-8)
        return optimizer


import tiktoken


import os
import numpy as np



class DataLoaderLite:
    def __init__(self, B, T, bin_file):
        """
        Args:
            B (int): batch size
            T (int): sequence length
            bin_file (str): path to pre-created .bin file containing token IDs
        """
        self.B = B
        self.T = T

        if not os.path.exists(bin_file):
            raise FileNotFoundError(f"{bin_file} not found. You must create it first.")

        # Memory-map the .bin file
        self.tokens = np.memmap(bin_file, dtype=np.uint16, mode="r")
        self.num_tokens = len(self.tokens)
        self.current_position = 0
        self.grad_accum_steps = None
        self.steps_needed = None

        print(f"Loaded {bin_file} with {self.num_tokens} tokens "
              f"({self.num_tokens // (B*T)} batches, streaming mode)")

    def set_grad_accum(self, grad_accum_steps):
        self.grad_accum_steps = grad_accum_steps
        tokens_per_step = self.B * self.T * self.grad_accum_steps
        self.steps_needed = math.ceil(self.num_tokens / tokens_per_step)
        print(f"With grad_accum_steps={grad_accum_steps}, steps_needed={self.steps_needed}")

    def reset(self):
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        start = self.current_position
        end = start + (B * T + 1)

        if end > self.num_tokens:
            self.reset()
            start = 0
            end = B * T + 1

        buf = torch.from_numpy(self.tokens[start:end].astype(np.int64))
        x = buf[:-1].view(B, T).contiguous()
        y = buf[1:].view(B, T).contiguous()

        self.current_position = start + (B * T)
        return x, y





        
import time



from train_gpt2 import GPT, GPTConfig, DataLoaderLite

if __name__ == "__main__":

    enc = tiktoken.get_encoding('gpt2')

    total_batch_size = 6144  # total tokens per batch (across all micro-batches)
    B = 12
    T = 256  # try reducing to 512 or 1024 if you get OOM

    train_loader = DataLoaderLite(B=12, T=256, bin_file="train.bin")
    val_loader = DataLoaderLite(B=1, T=256, bin_file="val.bin")

    grad_accum_steps = total_batch_size // (B * T)
    train_loader.set_grad_accum(grad_accum_steps)
    val_loader.set_grad_accum(1)  # usually no grad accumulation for val

    max_steps = train_loader.steps_needed  # will cover entire dataset

    # print("Starting model load...")
    # torch.set_float32_matmul_precision('high')

    # # Data loaders


    # # Model
    # model = GPT(GPTConfig(vocab_size=50257))
    # model.load_state_dict(torch.load("gpt_model.pth", map_location='cuda'))  # adjust map_location
    # checkpoint = torch.load("gpt_checkpoint.pth")
    # model.load_state_dict(checkpoint['model'],map_location='cuda')
    # model.to('cuda')
    # model.eval()
    
    print("Starting model load...")
torch.set_float32_matmul_precision('high')

# Initialize model
model = GPT(GPTConfig(vocab_size=50257))
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=3e-4, device='cuda')

# Load checkpoint if it exists
checkpoint_path = "gpt_checkpoint.pth"
if os.path.exists(checkpoint_path):
    print(f"Found checkpoint at {checkpoint_path}, loading weights...")
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
else:
    print("No checkpoint found, starting training from scratch.")

model.to('cuda')
model.train()


    # Learning rate schedule
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 10
    

    def get_lr(it):
        if it < warmup_steps:
            return max_lr * (it + 1) / warmup_steps
        elif it <= max_steps:
            decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return min_lr + coeff * (max_lr - min_lr)
        else:
            return min_lr

    # Optimizer
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=3e-4, device='cuda')
    optimizer.load_state_dict(checkpoint['optimizer'])
    eval_interval = 100  # run validation every 100 steps
    eval_iters = 20      # average over 20 batches

    @torch.no_grad()
    def estimate_loss(train_loader, val_loader, model, eval_iters=20):
        model.eval()
        out = {}
        for split, loader in [('train', train_loader), ('val', val_loader)]:
            losses = []
            for _ in range(eval_iters):
                x, y = loader.next_batch()
                x, y = x.to('cuda'), y.to('cuda')

                B, T = x.shape
                micro_batch_size = 8
                micro_losses = []

                for i in range(0, B, micro_batch_size):
                    xb = x[i:i + micro_batch_size]
                    yb = y[i:i + micro_batch_size]
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        logits, loss = model(xb, yb)
                    micro_losses.append(loss.item())

                losses.append(sum(micro_losses) / len(micro_losses))

                # Free memory
                del x, y, logits, loss, xb, yb
                torch.cuda.empty_cache()

            out[split] = sum(losses) / len(losses)
        model.train()
        return out

    # Training loop
    for step in range(max_steps):
        start = time.time()

        optimizer.zero_grad()
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to('cuda'), y.to('cuda')
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.step()
        torch.cuda.synchronize()
        end = time.time()
        print(f" step{step} loss is {loss.item()} --- Time taken: {end - start:.6f} s, "
              f"Tokens/sec = {(train_loader.B * train_loader.T) / (end - start)}, lr={lr}, norm={norm}")

        # Validation
        if step % eval_interval == 0 or step == max_steps - 1:
            losses = estimate_loss(train_loader, val_loader, model, eval_iters)
            print(f">>> Eval at step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Save model
    # torch.save(model.state_dict(), "gpt_model.pth")
    torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict()
}, "gpt_checkpoint.pth")



  

    print("Model parameters saved to gpt_model.pth")

    # ------------------- Inference -------------------
    num_return_sequence = 5
    max_length = 30
    context_window = 128

    prompt = "Once upon a time,"
    tokens = torch.tensor(enc.encode(prompt), dtype=torch.long).unsqueeze(0)
    tokens = tokens.cpu()
    model.cpu()
    model.eval()

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
