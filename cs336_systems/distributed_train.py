"""
---
2025.10
OS: Ubuntu Linux
GPU: NVIDIA-GPU PCIE X 4
device: GPU0 GPU1 GPU2 GPU3
Shell: bash
Improvement: torch.compile() FlashAttention DDP OptimizerStateShare
Our Impl: Use torch.nn.parallel.DistributedDataParallel...

TODO: I only used a single rtx4090 and did not write a distributed strategy.
I will complete this after I reduce the loss to below 3.3.
"""
# ass1
import os
import sys
import json
import time
import glob
import yaml
from cs336_basics.data import load
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from cs336_basics.data import get_batch, load, save
from cs336_basics.tokenizer import Tokenizer, train_bpe, PAT_GPT2, PAT_SPECIAL_TOKEN
from cs336_basics.modules.layers import TransformerLM
from cs336_basics.modules.loss import CrossEntropyLoss
from cs336_basics.modules.activation import GLU, Softmax
from cs336_basics.modules.optimizer import SGD, AdamW, compute_lr, gradient_cliping
from torch.utils.tensorboard import SummaryWriter
# ass2
from tokenizers import Tokenizer as HFTokenizer
# Only Test
import wandb
from ddp_model import DDPIndividualParameters, BucketDDPIndividualParameters
from optimizer_share import OptimizerStateShare
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim


def train():
    # 1. Load Config
    print("--- Loading Configuration ---")
    with open('config.yaml', 'r') as f:
        load = yaml.safe_load(f)
        
    model_args = load['model_args']
    training_args = load['training_args']
    data_args = load['data_args']
    
    os.makedirs(data_args['checkpoint_dir'], exist_ok=True)
    
    # 2. Initial
    print("--- Initializing ---")
    device = torch.device(training_args['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    hf_tokenizer = HFTokenizer.from_file(data_args['vocab_path'])

    tokenizer = Tokenizer(
        vocab={token: idx for token, idx in hf_tokenizer.get_vocab().items()},
        merges=data_args['merges_path']
    )

    model_args['vocab_size'] = len(tokenizer.vocab)
    print(f"Tokenizer loaded. Vocab size: {model_args['vocab_size']}")
    
    model = torch.compile(TransformerLM(**model_args)).to(device)
    print(f"Model created with {model.get_num_params():,} parameters.")
    
    optimizer = AdamW(model.parameters(), lr=training_args['learning_rate'])
    
    loss_init = CrossEntropyLoss()
    
    # 3. Load data
    print("--- Loading Data with np.memmap ---")
    train_data = np.memmap(data_args['train_data_path'], dtype=np.uint16, model='r')
    valid_data = np.memmap(data_args['valid_data_path'], dtype=np.uint16, model='r')
    print(f"Train data tokens: {len(train_data):,}, Val data tokens: {len(valid_data):,}")

    # 4. Resume training
    start_iter = 0
    if data_args['resume_from_checkpoint']:
        print(f"Resuming training from {data_args['resume_from_checkpoint']}")
        start_iter = load(data_args['resume_from_checkpoint'], model, optimizer)
    
    # 5. Evaluation Function
    @torch.no_grad()
    def evaluate():
        model.eval()
        valid_loss = 0
        eval_iters = 100
        for _ in range(eval_iters):
            x, y = get_batch(valid_data, training_args['batch_size'], model_args['context_length'], device)
            logits = model(x)
            loss = loss_init(logits.view(-1, model_args['vocab_size']), y.view(-1))
            valid_loss += loss.item()
        model.train()
        return valid_loss / eval_iters  

    # 6. Begin Training Loop
    ckpts = glob.glob(os.path.join(data_args['checkpoint_dir'], "model_iter_*.pt"))
    if ckpts:
        latest_ckpt = max(ckpts, key=os.path.getctime)
        start_iter = load(latest_ckpt, model, optimizer) + 1
    else:
        print("From Scratch Training")
    start_iter = 0
    
    writer = SummaryWriter(log_dir="runs/llm_run")

    train_losses = []
    val_losses = []
    iterations = []
    eval_iterations = []

    print("--- Starting Training Loop ---")
    t0 = time.time()

    for iter_num in range(start_iter, training_args['max_iters']):

        lr = compute_lr(
            iter_num,
            training_args['learning_rate'],
            training_args['min_lr'],
            training_args['warmup_steps'],
            training_args['lr_decay_steps']
        )
    
        for pg in optimizer.param_groups:
            pg['lr'] = lr


        inputs, targets = get_batch(
            train_data,
            training_args['batch_size'],
            model_args['context_length'],
            device
        )
    
        logits = model(inputs)

        loss = loss_init(
            logits.view(-1, model_args['vocab_size']),
            targets.view(-1)
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        total_norm = gradient_cliping(
            model.parameters(),
            training_args['gradient_clip_val']
        )
        optimizer.step()

        train_losses.append(loss.item())
        iterations.append(iter_num)

        writer.add_scalar("Loss/train", loss.item(), iter_num)
        writer.add_scalar("LR", lr, iter_num)
        writer.add_scalar("GradNorm/total", total_norm, iter_num)

        if iter_num % 10 == 0:
            t1 = time.time()
            print(
                f"Iter {iter_num}/{training_args['max_iters']}, "
                f"Train Loss: {loss.item():.4f}, "
                f"LR: {lr:.6f}, "
                f"GradNorm: {total_norm:.4f}, "
                f"Time: {(t1-t0)*1000:.1f}ms"
            )
            t0 = t1

        if iter_num > 0 and iter_num % training_args['eval_interval'] == 0:
            val_loss = evaluate()
            val_losses.append(val_loss)
            eval_iterations.append(iter_num)

            writer.add_scalar("Loss/val", val_loss, iter_num)

            print(f"--- Eval at iter {iter_num}: Val Loss: {val_loss:.4f} ---")
            checkpoint_path = os.path.join(
                data_args['checkpoint_dir'],
                f"model_iter_{iter_num}.pt"
            )
            save(model, optimizer, iter_num, checkpoint_path)

    writer.close()
    print("Training complete! Logs written to runs/llm_run")
            
if __name__ == '__main__':
    train()