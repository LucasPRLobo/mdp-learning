# probe.py
# TV-distance probe: compare model's next-state distribution against the true MDP
# transition distribution, using real trace prefixes as context.
import argparse
import json
import pickle
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from gridworld import build_vocab_with_pad, random_mdp, tokenize_trace
from model import GridworldTransformer


def tv(p, q):
    keys = set(p.keys()) | set(q.keys())
    return 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in keys)


def build_empirical_counts(dataset):
    empirical = defaultdict(Counter)
    for entry in dataset['traces']:
        mdp_idx = entry['mdp_idx']
        for step in entry['trace']:
            state, action, _, next_state = step
            empirical[(mdp_idx, tuple(state), action)][tuple(next_state)] += 1
    return empirical


@torch.no_grad()
def model_predict_with_real_context(model, mdp_idx, state, action, vocab, device,
                                    mdp_meta, tokenized, mdp_indices):
    """
    Find a real trace from this MDP that visits `state`, take the prefix ending at
    that state, append `action`, then marginalize the model's predictions over
    reward, next_state_x, next_state_y. Returns {(x, y): prob} or None.
    """
    model.eval()
    pad_id = vocab['<pad>']
    context_length = model.context_length

    target_state_x = vocab[f'coord_{state[0]}']
    target_state_y = vocab[f'coord_{state[1]}']

    prefix_tokens = None
    for tokens, mdp_i in zip(tokenized, mdp_indices):
        if mdp_i != mdp_idx:
            continue
        for step_start in range(0, len(tokens) - 6, 6):
            if tokens[step_start] == target_state_x and tokens[step_start + 1] == target_state_y:
                prefix_tokens = tokens[:step_start + 2] + [vocab[f'action_{action}']]
                break
        if prefix_tokens is not None:
            break

    if prefix_tokens is None or len(prefix_tokens) >= context_length:
        return None

    padded = prefix_tokens + [pad_id] * (context_length - 1 - len(prefix_tokens))
    input_ids = torch.tensor([padded[:context_length - 1]], dtype=torch.long, device=device)
    logits = model(input_ids)
    reward_probs = F.softmax(logits[0, len(prefix_tokens) - 1, :], dim=-1)

    distribution = {}
    for reward_name in ['reward_step', 'reward_goal']:
        r_id = vocab[reward_name]
        p_r = reward_probs[r_id].item()
        if p_r < 1e-6:
            continue

        ext_r = prefix_tokens + [r_id]
        padded_r = ext_r + [pad_id] * (context_length - 1 - len(ext_r))
        logits_r = model(torch.tensor([padded_r[:context_length - 1]], dtype=torch.long, device=device))
        next_x_probs = F.softmax(logits_r[0, len(ext_r) - 1, :], dim=-1)

        for x in range(mdp_meta['size_x']):
            x_id = vocab[f'coord_{x}']
            p_x = next_x_probs[x_id].item()
            if p_x < 1e-6:
                continue

            ext_x = ext_r + [x_id]
            padded_x = ext_x + [pad_id] * (context_length - 1 - len(ext_x))
            logits_x = model(torch.tensor([padded_x[:context_length - 1]], dtype=torch.long, device=device))
            next_y_probs = F.softmax(logits_x[0, len(ext_x) - 1, :], dim=-1)

            for y in range(mdp_meta['size_y']):
                y_id = vocab[f'coord_{y}']
                p_y = next_y_probs[y_id].item()
                joint = p_r * p_x * p_y
                if joint > 1e-6:
                    key = (x, y)
                    distribution[key] = distribution.get(key, 0) + joint

    total = sum(distribution.values())
    if total > 0:
        distribution = {k: v / total for k, v in distribution.items()}
    return distribution


def make_model(size, vocab_size, context_length, pad_id):
    if size == 'small':
        return GridworldTransformer(
            vocab_size=vocab_size, context_length=context_length,
            d_model=64, nhead=4, num_layers=2, dim_feedforward=128, pad_id=pad_id,
        )
    return GridworldTransformer(
        vocab_size=vocab_size, context_length=context_length,
        d_model=128, nhead=8, num_layers=4, dim_feedforward=256, pad_id=pad_id,
    )


def get_val_mdp_indices(num_mdps):
    # Mirrors train.py: 20% with floor of 2, fixed split seed.
    val_size = max(2, num_mdps // 5)
    split_rng = np.random.default_rng(0)
    return set(split_rng.choice(list(range(num_mdps)), size=val_size, replace=False).tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--size', choices=['small', 'large'], required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--context_length', type=int, default=128)
    parser.add_argument('--min_count', type=int, default=30,
                        help='Minimum empirical visit count to include a (state, action) probe')
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open(args.data_path, 'rb') as f:
        dataset = pickle.load(f)

    vocab = build_vocab_with_pad(max_grid_size=5, actions=['up', 'down', 'left', 'right'])
    pad_id = vocab['<pad>']
    tokenized = [tokenize_trace(entry['trace'], vocab) for entry in dataset['traces']]
    mdp_indices = [entry['mdp_idx'] for entry in dataset['traces']]

    val_set = get_val_mdp_indices(len(dataset['mdps']))
    empirical = build_empirical_counts(dataset)

    model = make_model(args.size, len(vocab), args.context_length, pad_id).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    results = []
    for mdp_idx, mdp_meta in enumerate(dataset['mdps']):
        mdp = random_mdp(seed=mdp_meta['mdp_seed'])
        for state in mdp.states:
            if state in mdp.terminal_states:
                continue
            for action in mdp.actions:
                counts = empirical[(mdp_idx, state, action)]
                total = sum(counts.values())
                if total < args.min_count:
                    continue

                true_dist = mdp.stochastic_transitions(state, action)
                emp_dist = {s: c / total for s, c in counts.items()}
                model_dist = model_predict_with_real_context(
                    model, mdp_idx, state, action, vocab, device, mdp_meta,
                    tokenized, mdp_indices,
                )
                if model_dist is None:
                    continue

                results.append({
                    'mdp_idx': mdp_idx,
                    'state': list(state),
                    'action': action,
                    'n': total,
                    'is_train_mdp': mdp_idx not in val_set,
                    'tv_true_emp': tv(true_dist, emp_dist),
                    'tv_true_model': tv(true_dist, model_dist),
                    'tv_emp_model': tv(emp_dist, model_dist),
                })

    train_tvs = [r['tv_true_model'] for r in results if r['is_train_mdp']]
    val_tvs = [r['tv_true_model'] for r in results if not r['is_train_mdp']]
    summary = {
        'checkpoint': args.checkpoint,
        'data_path': args.data_path,
        'size': args.size,
        'num_probed': len(results),
        'num_train_probes': len(train_tvs),
        'num_val_probes': len(val_tvs),
        'mean_tv_true_emp': float(np.mean([r['tv_true_emp'] for r in results])) if results else None,
        'mean_tv_true_model_train': float(np.mean(train_tvs)) if train_tvs else None,
        'mean_tv_true_model_val': float(np.mean(val_tvs)) if val_tvs else None,
    }

    with open(args.output, 'w') as f:
        json.dump({'summary': summary, 'results': results}, f, indent=2)

    print(f"Probed {len(results)} transitions ({len(train_tvs)} train, {len(val_tvs)} val)")
    if train_tvs:
        print(f"Train TV(true, model): {summary['mean_tv_true_model_train']:.4f}")
    if val_tvs:
        print(f"Val   TV(true, model): {summary['mean_tv_true_model_val']:.4f}")
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
