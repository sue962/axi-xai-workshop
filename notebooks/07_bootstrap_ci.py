'''
07 — Bootstrap 95% CI for the consolidation-gap ratio.

Addresses professor feedback (round 2, item 3): "Quantify uncertainty —
show error bars or bootstrap CI on the 5.37x finding."

Re-runs sequential CFL across 3 seeds (42, 123, 7) with per-round gap
logging, then runs a clustered bootstrap (resample rounds within each
seed, average across seeds) to get a 95% CI on the mean transition/
steady ratio.

Outputs:
    data/processed/bootstrap_per_round.csv  — (seed, round, gap, is_transition)
    data/processed/bootstrap_ci.txt          — CI summary
'''

from pathlib import Path
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

NOTEBOOK_DIR = Path.cwd()
PROJECT_ROOT = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == 'notebooks' else NOTEBOOK_DIR

DATA_DIR = PROJECT_ROOT / 'data/processed'
OUT_DIR  = DATA_DIR

N_CLIENTS    = 4
HIDDEN       = 64
LOCAL_EPOCHS = 1
BATCH_SIZE   = 256
LR           = 1e-3
ATTACK_TYPES = ['DoS', 'Fuzzy', 'RPM', 'gear']
N_ROUNDS_SEQ = 30
SEEDS        = [42, 123, 7]
N_BOOTSTRAP  = 5000
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

print(f'device       = {DEVICE}')
print(f'seeds        = {SEEDS}')
print(f'bootstrap N  = {N_BOOTSTRAP}')


# -------------------------- shared model / utils (matches 06) --------------------------

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN // 2), nn.ReLU(),
            nn.Linear(HIDDEN // 2, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_loader(X, y, batch=BATCH_SIZE, shuffle=True):
    return DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                      batch_size=batch, shuffle=shuffle)


def train_local(model, X, y, epochs=LOCAL_EPOCHS, lr=LR):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    fn  = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        for xb, yb in make_loader(X, y):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            fn(model(xb), yb).backward()
            opt.step()
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def average_weights(states, weights):
    total = sum(weights)
    weights = [w / total for w in weights]
    avg = copy.deepcopy(states[0])
    for k in avg:
        avg[k] = sum(w * sd[k] for w, sd in zip(weights, states))
    return avg


@torch.no_grad()
def predict_proba(model, X, batch=1024):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        out.append(torch.sigmoid(model(torch.from_numpy(X[i:i + batch]).to(DEVICE))).cpu().numpy())
    return np.concatenate(out)


def eval_per_source(model, X, y, src):
    yhat = (predict_proba(model, X) > 0.5).astype(int)
    out = {'overall_acc': float((yhat == y).mean())}
    for atk in ATTACK_TYPES:
        m = (src == atk) & (y == 1)
        out[f'recall_{atk}'] = float('nan') if m.sum() == 0 else float(yhat[m].mean())
    return out


def load_and_split(seed):
    df = pd.read_csv(DATA_DIR / 'can_combined.csv')
    feat_cols = ['can_id', 'dlc'] + [f'b{i}' for i in range(8)]
    train_df, test_df = train_test_split(
        df, test_size=0.20, random_state=seed, stratify=df['source']
    )
    scaler = StandardScaler().fit(train_df[feat_cols].values)
    X_train = scaler.transform(train_df[feat_cols].values).astype(np.float32)
    y_train = train_df['label'].values.astype(np.float32)
    X_test  = scaler.transform(test_df[feat_cols].values).astype(np.float32)
    y_test  = test_df['label'].values.astype(np.float32)
    src_test  = test_df['source'].values
    src_train = train_df['source'].values
    return X_train, y_train, src_train, X_test, y_test, src_test


def partition_non_iid(X_train, y_train, src_train, seed):
    rng = np.random.default_rng(seed)
    normal_idx = np.where(src_train == 'Normal')[0]
    rng.shuffle(normal_idx)
    normal_chunks = np.array_split(normal_idx, N_CLIENTS)
    clients_X, clients_y, clients_src = [], [], []
    for i, atk in enumerate(ATTACK_TYPES):
        own = np.where(src_train == atk)[0]
        other_mask = np.isin(src_train, [a for a in ATTACK_TYPES if a != atk])
        other_all = np.where(other_mask)[0]
        other = rng.choice(other_all, size=int(len(other_all) * 0.10), replace=False)
        idx = np.concatenate([normal_chunks[i], own, other])
        rng.shuffle(idx)
        clients_X.append(X_train[idx])
        clients_y.append(y_train[idx].copy())
        clients_src.append(src_train[idx])
    return clients_X, clients_y, clients_src


# -------------------------- sequential CFL with per-round gap logging --------------------------

def run_sequential_cfl_detailed(seed):
    """Returns DataFrame with columns: seed, round, gap, is_transition."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, y_train, src_train, X_test, y_test, src_test = load_and_split(seed)
    clients_X, clients_y, clients_src = partition_non_iid(X_train, y_train, src_train, seed)

    rounds_per_phase = N_ROUNDS_SEQ // len(ATTACK_TYPES)
    transition_rounds = {1, 1 + rounds_per_phase,
                         1 + 2 * rounds_per_phase, 1 + 3 * rounds_per_phase}

    global_model = MLP().to(DEVICE)
    slow_rows, fast_rows = [], []

    for r in range(1, N_ROUNDS_SEQ + 1):
        phase = min((r - 1) // rounds_per_phase, len(ATTACK_TYPES) - 1)
        active_task = ATTACK_TYPES[phase]
        states, weights = [], []
        for i in range(N_CLIENTS):
            mask = (clients_src[i] == 'Normal') | (clients_src[i] == active_task)
            Xi, yi = clients_X[i][mask], clients_y[i][mask]
            if len(Xi) == 0:
                continue
            local = MLP().to(DEVICE)
            local.load_state_dict(global_model.state_dict())
            sd = train_local(local, Xi, yi)
            fev = eval_per_source(local, X_test, y_test, src_test)
            for atk in ATTACK_TYPES:
                fast_rows.append({'round': r, 'client_id': i, 'attack': atk,
                                  'fast_recall': fev[f'recall_{atk}']})
            states.append(sd)
            weights.append(len(Xi))
        global_model.load_state_dict(average_weights(states, weights))
        sev = eval_per_source(global_model, X_test, y_test, src_test)
        slow_rows.append({'round': r, **sev})

    slow_df = pd.DataFrame(slow_rows)
    fast_df = pd.DataFrame(fast_rows)

    rows = []
    for r in range(1, N_ROUNDS_SEQ + 1):
        slow_r = slow_df[slow_df['round'] == r].iloc[0]
        fast_r = fast_df[fast_df['round'] == r]
        gaps = []
        for _, row in fast_r.iterrows():
            gap = abs(row['fast_recall'] - slow_r[f'recall_{row["attack"]}'])
            gaps.append(gap)
        rows.append({
            'seed': seed,
            'round': r,
            'gap':     float(np.mean(gaps)),
            'max_gap': float(np.max(gaps)),
            'is_transition': int(r in transition_rounds),
        })
    return pd.DataFrame(rows)


# -------------------------- main --------------------------

if __name__ == '__main__':
    all_rows = []
    print('\n' + '=' * 70)
    print('Step 1 — sequential CFL x 3 seeds (per-round gap logging)')
    print('=' * 70)
    for seed in SEEDS:
        print(f'\n[seed={seed}] ...', flush=True)
        df = run_sequential_cfl_detailed(seed)
        all_rows.append(df)
        t = df[df['is_transition'] == 1]['gap']
        s = df[df['is_transition'] == 0]['gap']
        print(f'  transition mean = {t.mean():.4f}  (n={len(t)})')
        print(f'  steady     mean = {s.mean():.4f}  (n={len(s)})')
        print(f'  per-seed ratio  = {t.mean() / s.mean():.2f}x')

    per_round = pd.concat(all_rows, ignore_index=True)
    per_round.to_csv(OUT_DIR / 'bootstrap_per_round.csv', index=False)
    print(f'\nSaved: {OUT_DIR / "bootstrap_per_round.csv"}')

    # ---------- Clustered bootstrap on the mean-of-per-seed ratio ----------
    print('\n' + '=' * 70)
    print(f'Step 2 — Clustered bootstrap ({N_BOOTSTRAP} iterations)')
    print('=' * 70)
    rng = np.random.default_rng(42)

    t_by_seed = {s: per_round[(per_round['seed'] == s) & (per_round['is_transition'] == 1)]['gap'].values
                 for s in SEEDS}
    s_by_seed = {s: per_round[(per_round['seed'] == s) & (per_round['is_transition'] == 0)]['gap'].values
                 for s in SEEDS}

    per_seed_ratios = {s: float(t_by_seed[s].mean() / s_by_seed[s].mean()) for s in SEEDS}
    point_ratio = float(np.mean(list(per_seed_ratios.values())))

    boot_ratios, boot_t_means, boot_s_means = [], [], []
    for _ in range(N_BOOTSTRAP):
        seed_ratios, seed_t, seed_s = [], [], []
        for s in SEEDS:
            tr = rng.choice(t_by_seed[s], size=len(t_by_seed[s]), replace=True)
            sr = rng.choice(s_by_seed[s], size=len(s_by_seed[s]), replace=True)
            tm, sm = tr.mean(), sr.mean()
            seed_t.append(tm)
            seed_s.append(sm)
            if sm > 0:
                seed_ratios.append(tm / sm)
        if seed_ratios:
            boot_ratios.append(np.mean(seed_ratios))
            boot_t_means.append(np.mean(seed_t))
            boot_s_means.append(np.mean(seed_s))

    boot_ratios  = np.array(boot_ratios)
    boot_t_means = np.array(boot_t_means)
    boot_s_means = np.array(boot_s_means)

    def ci(arr, alpha=0.05):
        return float(np.percentile(arr, 100 * alpha / 2)), float(np.percentile(arr, 100 * (1 - alpha / 2)))

    r_lo, r_hi = ci(boot_ratios)
    t_lo, t_hi = ci(boot_t_means)
    s_lo, s_hi = ci(boot_s_means)

    # transition / steady point means (averaged across seeds)
    t_point = float(np.mean([t_by_seed[s].mean() for s in SEEDS]))
    s_point = float(np.mean([s_by_seed[s].mean() for s in SEEDS]))

    lines = []
    lines.append('=' * 70)
    lines.append('BOOTSTRAP 95% CI - consolidation-gap statistics')
    lines.append(f'  resamples = {N_BOOTSTRAP}, seeds = {SEEDS}')
    lines.append(f'  method    = clustered bootstrap (resample rounds within seed,')
    lines.append(f'              then average per-seed ratios)')
    lines.append('=' * 70)
    lines.append('')
    lines.append('Per-seed ratios (transition / steady):')
    for s in SEEDS:
        lines.append(f'  seed {s:3d}: {per_seed_ratios[s]:.2f}x')
    lines.append('')
    lines.append('Transition / steady ratio (mean across seeds):')
    lines.append(f'  point estimate : {point_ratio:.2f}x')
    lines.append(f'  95% CI         : [{r_lo:.2f}, {r_hi:.2f}]')
    lines.append('')
    lines.append('Transition-round mean gap:')
    lines.append(f'  point estimate : {t_point:.4f}')
    lines.append(f'  95% CI         : [{t_lo:.4f}, {t_hi:.4f}]')
    lines.append('')
    lines.append('Steady-round mean gap:')
    lines.append(f'  point estimate : {s_point:.4f}')
    lines.append(f'  95% CI         : [{s_lo:.4f}, {s_hi:.4f}]')

    summary = '\n'.join(lines)
    print('\n' + summary)
    (OUT_DIR / 'bootstrap_ci.txt').write_text(summary)
    print(f'\nSaved: {OUT_DIR / "bootstrap_ci.txt"}')
