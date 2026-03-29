import os
import time
import glob
import math
import yaml
import random
import wandb
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from cs336_basics.data import get_batch, load, save
from cs336_basics.modules.layers import TransformerLM
from cs336_basics.modules.loss import CrossEntropyLoss
from cs336_basics.modules.optimizer import AdamW, compute_lr, gradient_cliping
from cs336_basics.tokenizer import Tokenizer
from tokenizers import Tokenizer as HFTokenizer


def train_loop():
    """Helper/Config"""
    print("--- Loading Configuration ---")
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    model_args = cfg['model_args']
    training_args = cfg['training_args']
    data_args = cfg['data_args']

    os.makedirs(data_args['checkpoint_dir'], exist_ok=True)

    device = torch.device(training_args.get('device', 'cuda') if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    seed = training_args.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
        if training_args.get('cudnn_benchmark', True):
            torch.backends.cudnn.benchmark = True
        if training_args.get('cudnn_deterministic', False):
            torch.backends.cudnn.deterministic = True

    """Tokenizer/Data"""
    hf_tokenizer = HFTokenizer.from_file(data_args['vocab_path'])
    tokenizer = Tokenizer(
        vocab={token: idx for token, idx in hf_tokenizer.get_vocab().items()},
        merges=data_args['merges_path']
    )
    model_args['vocab_size'] = len(tokenizer.vocab)
    print(f"Tokenizer loaded. Vocab size: {model_args['vocab_size']}")

    """Model Init"""
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.normal_(module.weight, mean=0.0, std=training_args.get('init_std', 0.02))
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=training_args.get('init_std', 0.02))
        elif isinstance(module, nn.LayerNorm):
            if getattr(module, 'weight', None) is not None:
                nn.init.ones_(module.weight)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def residual_projection_rescale(model, num_layers):
        factor = math.sqrt(2.0 * max(1, num_layers))
        for name, p in model.named_parameters():
            if name.endswith('c_proj.weight') or '.c_proj.' in name:
                with torch.no_grad():
                    if p.dtype.is_floating_point:
                        p.mul_(1.0 / factor)

    def try_weight_tying(model):
        emb = None
        head = None
        for attr in ('token_embedding', 'tok_embeddings', 'wte', 'embed_tokens', 'embedding'):
            if hasattr(model, attr):
                emb = getattr(model, attr)
                break
        for attr in ('lm_head', 'head', 'output_head', 'decoder'):
            if hasattr(model, attr):
                head = getattr(model, attr)
                break
        if emb is None:
            for n, m in model.named_modules():
                if isinstance(m, nn.Embedding):
                    emb = m
                    break
        if head is None:
            for n, m in model.named_modules():
                if isinstance(m, nn.Linear):
                    head = m
        if isinstance(emb, nn.Embedding) and isinstance(head, nn.Linear):
            try:
                head_weight = getattr(head, 'weight', None)
                if head_weight is not None and emb.weight.shape == head_weight.shape:
                    emb.weight = head.weight
                    print("Weight tying applied: embedding <-> lm_head")
                else:
                    if emb.weight.shape[1] == head.weight.shape[0] and emb.weight.shape[0] == head.weight.shape[1]:
                        print("Found embedding and head but shapes do not match for direct tying; skipping tying.")
            except Exception as e:
                print("Weight tying failed (ignored):", e)

    """Model Build"""
    print("--- Building model (uncompiled) ---")
    model = TransformerLM(**model_args).to(device)
    model.apply(_init_weights)

    num_layers = None
    if hasattr(model, 'num_layers'):
        num_layers = getattr(model, 'num_layers')
    elif hasattr(model, 'n_layer'):
        num_layers = getattr(model, 'n_layer')
    elif 'num_layers' in model_args:
        num_layers = model_args['num_layers']
    elif 'n_layer' in model_args:
        num_layers = model_args['n_layer']
    else:
        for name, module in model.named_children():
            if name in ('blocks', 'layers', 'transformer_blocks'):
                num_layers = sum(1 for _ in module.children())
                break
    if num_layers is None:
        num_layers = training_args.get('num_layers', 12)

    residual_projection_rescale(model, num_layers)
    try_weight_tying(model)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {num_params:,} parameters.")

    """Optimizer/Loss"""
    optimizer = AdamW(model.parameters(), lr=training_args['learning_rate'])
    loss_fn = CrossEntropyLoss()

    """Checkpoint resume"""
    start_iter = 0
    # if data_args.get('resume_from_checkpoint'):
    #     ckpt_path = data_args['resume_from_checkpoint']
    #     print(f"Resuming training from explicit checkpoint: {ckpt_path}")
    #     start_iter = load(ckpt_path, model, optimizer) or 0

    # ckpts = glob.glob(os.path.join(data_args['checkpoint_dir'], "model_iter_*.pt"))
    # if ckpts:
    #     latest_ckpt = max(ckpts, key=os.path.getctime)
    #     print(f"Found latest checkpoint: {latest_ckpt}")
    #     start_iter = load(latest_ckpt, model, optimizer) or start_iter
    # else:
    #     if start_iter == 0:
    print("From Scratch Training")

    """Compile"""
    if training_args.get('use_torch_compile', True) and hasattr(torch, 'compile'):
        try:
            print("Compiling model with torch.compile() ...")
            model = torch.compile(model)
        except Exception as e:
            print("torch.compile() failed or not beneficial; continuing without compile:", e)

    use_amp = training_args.get('use_amp', True) and device.type == 'cuda'
    amp_dtype = torch.bfloat16 if use_amp else None
    if use_amp:
        print("Enabled Mixed Precision (AMP) with torch.bfloat16 for Hopper Architecture.")
    amp_context = torch.amp.autocast('cuda', dtype=amp_dtype) if use_amp else torch.autocast('cpu', enabled=False)

    """Data"""
    print("--- Loading Data with np.memmap ---")
    train_data = np.memmap(data_args['train_data_path'], dtype=np.uint16, mode='r')
    valid_data = np.memmap(data_args['valid_data_path'], dtype=np.uint16, mode='r')
    print(f"Train tokens: {len(train_data):,}, Val tokens: {len(valid_data):,}")

    """Wandb"""
    use_wandb = training_args.get('use_wandb', True)
    if use_wandb:
        wandb.init(
            entity='lzq666amn-github',
            project="B200-40000-1",
            config={**model_args},
        )

    writer = SummaryWriter(log_dir=training_args.get('tensorboard_logdir', "runs/llm_run"))

    """Evaluate"""
    @torch.no_grad()
    def evaluate(iter_num):
        model.eval()
        valid_loss = 0.0
        if iter_num < 500:
            eval_iters = min(10, training_args.get('eval_max_iters', 20))
        elif iter_num < 5000:
            eval_iters = min(50, training_args.get('eval_max_iters', 100))
        else:
            eval_iters = training_args.get('eval_max_iters', 100)

        for _ in range(eval_iters):
            x, y = get_batch(valid_data, training_args['batch_size'], model_args['context_length'], device)
            with amp_context:
                logits = model(x)
                loss = loss_fn(logits.view(-1, model_args['vocab_size']), y.view(-1))
            valid_loss += loss.item()
        model.train()

        avg_loss = valid_loss / max(1, eval_iters)
        perplexity = math.exp(avg_loss)
        return avg_loss, perplexity

    """Training Loop"""
    grad_accum_steps = training_args.get('grad_accum_steps', 1)
    max_iters = training_args['max_iters']
    t0 = time.time()

    print("--- Starting Training Loop ---")
    train_losses, val_losses, iterations, eval_iterations = [], [], [], []

    for iter_num in range(start_iter, max_iters):
        lr = compute_lr(
            iter_num,
            training_args['learning_rate'],
            training_args.get('min_lr', 0.0),
            training_args.get('warmup_steps', 100),
            training_args.get('lr_decay_steps', max_iters)
        )
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        optimizer.zero_grad(set_to_none=True)
        micro_loss = 0.0
        for micro_step in range(grad_accum_steps):
            inputs, targets = get_batch(train_data, training_args['batch_size'], model_args['context_length'], device)
            with amp_context:
                logits = model(inputs)
                loss = loss_fn(logits.view(-1, model_args['vocab_size']), targets.view(-1))
                loss = loss / grad_accum_steps
            loss.backward()
            micro_loss += loss.item()

        total_norm = gradient_cliping(model.parameters(), training_args['gradient_clip_val'])
        optimizer.step()

        iter_loss = micro_loss * grad_accum_steps
        train_ppl = math.exp(iter_loss)

        train_losses.append(iter_loss)
        iterations.append(iter_num)

        writer.add_scalar("Loss/train", iter_loss, iter_num)
        writer.add_scalar("Perplexity/train", train_ppl, iter_num)
        writer.add_scalar("LR", lr, iter_num)
        writer.add_scalar("GradNorm/total", total_norm, iter_num)

        if use_wandb:
            wandb.log({
                "train/loss": iter_loss,
                "train/perplexity": train_ppl,
                "lr": lr,
                "grad_norm": total_norm,
                "iter": iter_num
            }, step=iter_num)

        if iter_num % training_args.get('log_interval', 10) == 0:
            t1 = time.time()
            print(
                f"Iter {iter_num}/{max_iters}, "
                f"Train Loss: {iter_loss:.4f}, "
                f"Train PPL: {train_ppl:.2f}, "
                f"LR: {lr:.6f}, "
                f"GradNorm: {total_norm:.4f}, "
                f"Time per {training_args.get('log_interval',10)} iters: {(t1-t0)*1000:.1f}ms"
            )
            t0 = t1

        if iter_num > 0 and iter_num % training_args.get('eval_interval', 1000) == 0:
            val_loss, val_ppl = evaluate(iter_num)
            val_losses.append(val_loss)
            eval_iterations.append(iter_num)

            writer.add_scalar("Loss/val", val_loss, iter_num)
            writer.add_scalar("Perplexity/val", val_ppl, iter_num)

            if use_wandb:
                wandb.log({"val/loss": val_loss, "val/perplexity": val_ppl, "iter": iter_num}, step=iter_num)

            print(f"--- Eval at iter {iter_num}: Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} ---")
            checkpoint_path = os.path.join(data_args['checkpoint_dir'], f"model_iter_{iter_num}.pt")
            try:
                save(model, optimizer, iter_num, checkpoint_path)
            except Exception:
                torch.save({
                    'iter': iter_num,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'rng_state': torch.get_rng_state(),
                }, checkpoint_path)
                print("Saved fallback checkpoint.")

    writer.close()
    if use_wandb:
        wandb.finish()
    print("Training complete! Logs written to", training_args.get('tensorboard_logdir', "runs/llm_run"))