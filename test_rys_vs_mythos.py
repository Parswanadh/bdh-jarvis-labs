import torch
import torch.nn as nn
from implementation.bdh_recurrent import BDHRecurrent, BDHRecurrentConfig
from implementation.multiscale_bdh import MultiScaleBDHConfig
import numpy as np

def analyze_state_stability(states):
    """Calculate variance and magnitude of states to check for explosion/vanishing."""
    # states is a list of state tuples: [layers [scales [C,C]]]
    all_vals = []
    for layer_states in states:
        for scale_state in layer_states:
            all_vals.append(scale_state.detach().cpu().numpy())

    combined = np.concatenate([v.flatten() for v in all_vals])
    return {
        'mean': np.mean(combined),
        'std': np.std(combined),
        'max': np.max(combined),
        'var': np.var(combined)
    }

def analyze_logits(logits):
    """Calculate entropy of logits to measure confidence."""
    probs = torch.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
    return {
        'avg_entropy': entropy.mean().item(),
        'min_entropy': entropy.min().item(),
        'max_entropy': entropy.max().item()
    }

def run_trial(name, config, input_ids):
    print(f"Running Trial {name}...")
    model = BDHRecurrent(config).to(device)
    model.eval()

    with torch.no_grad():
        # Use return_states=True and return_loop_metrics=True
        logits, states, metrics = model(input_ids, return_states=True, return_loop_metrics=True)

        state_stats = analyze_state_stability(states)
        logit_stats = analyze_logits(logits)

        return {
            'name': name,
            'loops': metrics['actual_loops'],
            'avg_depth': metrics['avg_loop_depth'],
            'state_var': state_stats['var'],
            'state_std': state_stats['std'],
            'entropy': logit_stats['avg_entropy'],
            'max_val': state_stats['max']
        }

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Shared Base Config
    base_cfg = MultiScaleBDHConfig(
        vocab_size=256, n_embd=256, n_layer=6, n_head=4, ffn_dim=1024
    )

    # Input: a random sequence of bytes
    input_ids = torch.randint(0, 256, (1, 32)).to(device)

    # Trial A: Vanilla BDH (1 loop, no fancy logic)
    cfg_a = BDHRecurrentConfig(
        base_config=base_cfg,
        max_loops=1,
        min_loops=1,
        use_learnable_decay=False,
        use_per_token_exit=False
    )

    # Trial B: Structural BDH-RD (Medium loops, loop embeddings, ACT)
    cfg_b = BDHRecurrentConfig(
        base_config=base_cfg,
        max_loops=4,
        min_loops=1,
        use_learnable_decay=False, # Keep fixed for structural test
        use_per_token_exit=True
    )

    # Trial C: BDH-RD Max (Deep loops, full recurrence)
    cfg_c = BDHRecurrentConfig(
        base_config=base_cfg,
        max_loops=12,
        min_loops=1,
        use_learnable_decay=False,
        use_per_token_exit=True
    )

    results = []
    results.append(run_trial("Trial A (Vanilla)", cfg_a, input_ids))
    results.append(run_trial("Trial B (Structural)", cfg_b, input_ids))
    results.append(run_trial("Trial C (Max Depth)", cfg_c, input_ids))

    print("\n" + "="*60)
    print(f"{'Trial':<<220} | {'Loops':<<88} | {'State Var':<<112} | {'Entropy':<<110}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<<220} | {r['loops']:<<88} | {r['state_var']:<<112.4f} | {r['entropy']:<<110.4f}")
    print("="*60)
