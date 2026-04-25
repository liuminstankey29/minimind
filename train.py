#!/usr/bin/env python3
"""
Training script for MiniMind language model.
Supports pretraining and supervised fine-tuning (SFT) modes.
"""

import os
import time
import math
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

from model.model import ModelConfig, MiniMindModel
from dataset.lm_dataset import PretrainDataset


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def train_epoch(model, loader, optimizer, scaler, device, epoch, args):
    """Run a single training epoch."""
    model.train()
    total_loss = 0.0
    start_time = time.time()

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Update learning rate
        global_step = epoch * len(loader) + step
        lr = get_lr(global_step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad()

        with autocast(enabled=args.use_amp):
            logits, loss = model(input_ids, labels=labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if step % args.log_interval == 0:
            elapsed = time.time() - start_time
            avg_loss = total_loss / (step + 1)
            print(
                f"Epoch {epoch} | Step {step}/{len(loader)} | "
                f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

    return total_loss / len(loader)


def parse_args():
    parser = argparse.ArgumentParser(description="Train MiniMind")
    # Data
    parser.add_argument("--data_path", type=str, default="dataset/pretrain_data.bin")
    parser.add_argument("--max_seq_len", type=int, default=512)
    # Model
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--vocab_size", type=int, default=32000)
    # Training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--use_amp", action="store_true", default=True)
    # Misc
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Dataset and DataLoader
    dataset = PretrainDataset(args.data_path, max_length=args.max_seq_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Model
    config = ModelConfig(
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
    )
    model = MiniMindModel(config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params / 1e6:.2f}M")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    scaler = GradScaler(enabled=args.use_amp)
    args.max_steps = args.epochs * len(loader)

    # Training loop
    for epoch in range(args.epochs):
        avg_loss = train_epoch(model, loader, optimizer, scaler, device, epoch, args)
        print(f"Epoch {epoch} complete. Avg loss: {avg_loss:.4f}")

        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(args.save_dir, f"minimind_epoch{epoch}.pt")
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "config": config},
                ckpt_path,
            )
            print(f"Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
