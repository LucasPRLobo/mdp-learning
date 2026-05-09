# train.py
import argparse
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from gridworld import build_vocab_with_pad, tokenize_trace
from model import GridworldTransformer

def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for full determinism, also:
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

class TraceDataset(Dataset):
    def __init__(self, tokenized_traces, mdp_indices, allowed_mdp_indices, context_length, pad_id):
        # filter to traces from allowed MDPs
        self.traces = [
            tokens for tokens, mdp_idx in zip(tokenized_traces, mdp_indices)
            if mdp_idx in allowed_mdp_indices and len(tokens) <= context_length
        ]
        self.context_length = context_length
        self.pad_id = pad_id
    
    def __len__(self):
        return len(self.traces)
    
    def __getitem__(self, idx):
        tokens = self.traces[idx]
        # pad to context_length
        padded = tokens + [self.pad_id] * (self.context_length - len(tokens))
        padded = torch.tensor(padded, dtype=torch.long)
        # input is everything except last position; target is everything except first
        input_ids = padded[:-1]
        target_ids = padded[1:]
        return input_ids, target_ids


def train_one_epoch(model, loader, criterion, optimizer, device, pad_id):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    
    for input_ids, target_ids in loader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        # Forward
        logits = model(input_ids)
        # logits: (batch, seq_len, vocab_size)
        # targets: (batch, seq_len)
        # Reshape for cross-entropy
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),  # (batch * seq_len, vocab_size)
            target_ids.reshape(-1)                 # (batch * seq_len,)
        )
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track loss (weighted by non-pad tokens for accurate average)
        non_pad = (target_ids != pad_id).sum().item()
        total_loss += loss.item() * non_pad
        total_tokens += non_pad
    
    return total_loss / total_tokens

@torch.no_grad()
def evaluate(model, loader, criterion, device, pad_id):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    for input_ids, target_ids in loader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        logits = model(input_ids)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1)
        )
        
        non_pad = (target_ids != pad_id).sum().item()
        total_loss += loss.item() * non_pad
        total_tokens += non_pad
    
    return total_loss / total_tokens
    

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--size', choices=['small', 'large'], required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--context_length', type=int, default=128)
    parser.add_argument('--data_path', type=str, default='dataset.pkl')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    
    set_all_seeds(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # --- Load dataset ---
    with open(args.data_path, 'rb') as f:
        dataset = pickle.load(f)
    
    # --- Build vocab and tokenize ---
    vocab = build_vocab_with_pad(max_grid_size=5, actions=['up', 'down', 'left', 'right'])
    pad_id = vocab['<pad>']
    
    tokenized = [tokenize_trace(entry['trace'], vocab) for entry in dataset['traces']]
    mdp_indices = [entry['mdp_idx'] for entry in dataset['traces']]
    
    # --- Train/val split (FIXED seed; does NOT depend on args.seed) ---
    num_mdps = len(dataset['mdps'])
    all_mdp_indices = list(range(num_mdps))
    val_size = max(2, num_mdps // 5)  # 20%, floor of 2
    split_rng = np.random.default_rng(0)  # fixed across all runs
    val_mdp_indices = sorted(split_rng.choice(all_mdp_indices, size=val_size, replace=False).tolist())
    train_mdp_indices = [i for i in all_mdp_indices if i not in val_mdp_indices]
    
    print(f"Train MDPs: {train_mdp_indices}")
    print(f"Val MDPs: {val_mdp_indices}")
    
    # --- Build datasets and loaders ---
    train_dataset = TraceDataset(
        tokenized, mdp_indices, set(train_mdp_indices), args.context_length, pad_id
    )
    val_dataset = TraceDataset(
        tokenized, mdp_indices, set(val_mdp_indices), args.context_length, pad_id
    )
    
    print(f"Train traces: {len(train_dataset)}")
    print(f"Val traces: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # --- Build model ---
    if args.size == 'small':
        model = GridworldTransformer(
            vocab_size=len(vocab), context_length=args.context_length,
            d_model=64, nhead=4, num_layers=2, dim_feedforward=128, pad_id=pad_id
        )
    else:
        model = GridworldTransformer(
            vocab_size=len(vocab), context_length=args.context_length,
            d_model=128, nhead=8, num_layers=4, dim_feedforward=256, pad_id=pad_id
        )
    model = model.to(device)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # --- Training ---
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, pad_id)
        val_loss = evaluate(model, val_loader, criterion, device, pad_id)
        print(f"Epoch {epoch+1}: train {train_loss:.4f}, val {val_loss:.4f}")
    
    # --- Save ---
    output = args.output or f'model_{args.size}_seed{args.seed}.pt'
    torch.save(model.state_dict(), output)
    print(f"Saved to {output}")


if __name__ == '__main__':
    main()