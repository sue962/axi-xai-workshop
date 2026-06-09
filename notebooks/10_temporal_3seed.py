'''
10 — 3-seed per-round per-client gap trajectory for Figure 7.

Runs the malicious-client detection (IID FedAvg, client 0 = label-flip 100%)
across the three seeds (42, 123, 7) used in 06_robustness.py, logging the
per-round per-client consolidation gap. Saves the long-form CSV and
re-plots Figure 7 as the per-client mean ± 1 std band across seeds.
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
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import set_paper_style
set_paper_style()

NOTEBOOK_DIR = Path.cwd()
PROJECT_ROOT = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == 'notebooks' else NOTEBOOK_DIR
DATA_DIR = PROJECT_ROOT / 'data/processed'
FIG_DIR  = PROJECT_ROOT / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEEDS         = [42, 123, 7]
N_CLIENTS     = 4
N_ROUNDS      = 12
LOCAL_EPOCHS  = 1
BATCH_SIZE    = 256
LR            = 1e-3
HIDDEN        = 64
ATTACK_TYPES  = ['DoS', 'Fuzzy', 'RPM', 'gear']
MAL_CLIENT    = 0
FLIP_RATE     = 1.0
DEVICE        = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN // 2), nn.ReLU(),
            nn.Linear(HIDDEN // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def loader(X, y, shuffle=True):
    return DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                      batch_size=BATCH_SIZE, shuffle=shuffle)


def train_local(model, X, y):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    fn  = nn.BCEWithLogitsLoss()
    for _ in range(LOCAL_EPOCHS):
        for xb, yb in loader(X, y):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); fn(model(xb), yb).backward(); opt.step()
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def avg_states(states, weights):
    total = sum(weights); weights = [w / total for w in weights]
    avg = copy.deepcopy(states[0])
    for k in avg:
        avg[k] = sum(w * sd[k] for w, sd in zip(weights, states))
    return avg


@torch.no_grad()
def predict_proba(model, X):
    model.eval(); out = []
    for i in range(0, len(X), 1024):
        out.append(torch.sigmoid(model(torch.from_numpy(X[i:i+1024]).to(DEVICE))).cpu().numpy())
    return np.concatenate(out)


def eval_per_source(model, X, y, src):
    yhat = (predict_proba(model, X) > 0.5).astype(int)
    out = {}
    for atk in ATTACK_TYPES:
        m = (src == atk) & (y == 1)
        out[f'recall_{atk}'] = float('nan') if m.sum() == 0 else float(yhat[m].mean())
    return out


def run_one_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    df = pd.read_csv(DATA_DIR / 'can_combined.csv')
    feat_cols = ['can_id', 'dlc'] + [f'b{i}' for i in range(8)]

    train_df, test_df = train_test_split(df, test_size=0.20, random_state=seed,
                                         stratify=df['source'])
    scaler  = StandardScaler().fit(train_df[feat_cols].values)
    X_train = scaler.transform(train_df[feat_cols].values).astype(np.float32)
    y_train = train_df['label'].values.astype(np.float32)
    X_test  = scaler.transform(test_df[feat_cols].values).astype(np.float32)
    y_test  = test_df['label'].values.astype(np.float32)
    src_test  = test_df['source'].values
    src_train = train_df['source'].values

    rng = np.random.default_rng(seed)
    normal_idx = np.where(src_train == 'Normal')[0]; rng.shuffle(normal_idx)
    normal_chunks = np.array_split(normal_idx, N_CLIENTS)

    clients_X, clients_y = [], []
    for i, atk in enumerate(ATTACK_TYPES):
        own = np.where(src_train == atk)[0]
        other_all = np.where(np.isin(src_train, [a for a in ATTACK_TYPES if a != atk]))[0]
        other = rng.choice(other_all, size=int(len(other_all) * 0.10), replace=False)
        idx = np.concatenate([normal_chunks[i], own, other]); rng.shuffle(idx)
        yi = y_train[idx].copy()
        if i == MAL_CLIENT:
            flip = rng.random(len(yi)) < FLIP_RATE
            yi[flip] = 1.0 - yi[flip]
        clients_X.append(X_train[idx]); clients_y.append(yi)

    global_model = MLP().to(DEVICE)
    slow_rows, fast_rows = [], []
    for r in range(1, N_ROUNDS + 1):
        states, weights = [], []
        for i in range(N_CLIENTS):
            local = MLP().to(DEVICE); local.load_state_dict(global_model.state_dict())
            sd = train_local(local, clients_X[i], clients_y[i])
            fev = eval_per_source(local, X_test, y_test, src_test)
            for atk in ATTACK_TYPES:
                fast_rows.append({'seed': seed, 'round': r, 'client_id': i,
                                  'attack': atk, 'fast_recall': fev[f'recall_{atk}']})
            states.append(sd); weights.append(len(clients_X[i]))
        global_model.load_state_dict(avg_states(states, weights))
        sev = eval_per_source(global_model, X_test, y_test, src_test)
        slow_rows.append({'seed': seed, 'round': r, **sev})

    slow_df = pd.DataFrame(slow_rows)
    fast_df = pd.DataFrame(fast_rows)

    # Per-round per-client gap = mean |fast_recall - slow_recall| over the 4 attacks
    gaps = []
    for cid in range(N_CLIENTS):
        for r in range(1, N_ROUNDS + 1):
            fsub = fast_df[(fast_df['client_id'] == cid) & (fast_df['round'] == r)]
            ssub = slow_df[slow_df['round'] == r].iloc[0]
            g = np.mean([abs(row['fast_recall'] - ssub[f'recall_{row["attack"]}'])
                         for _, row in fsub.iterrows()])
            gaps.append({'seed': seed, 'round': r, 'client_id': cid, 'gap': g})
    return pd.DataFrame(gaps)


def main():
    print(f'device: {DEVICE}, seeds: {SEEDS}')
    all_gaps = []
    for s in SEEDS:
        print(f'\n[seed {s}] running...')
        all_gaps.append(run_one_seed(s))
    gaps = pd.concat(all_gaps, ignore_index=True)
    gaps.to_csv(DATA_DIR / 'malicious_temporal_3seed.csv', index=False)
    print(f'\nsaved: {DATA_DIR / "malicious_temporal_3seed.csv"}  ({gaps.shape})')

    # Plot — mean ± 1 std band per client
    agg = (gaps.groupby(['client_id', 'round'])['gap']
                .agg(['mean', 'std']).reset_index())
    fig, ax = plt.subplots(figsize=(10, 4))
    for cid in range(N_CLIENTS):
        sub = agg[agg['client_id'] == cid].sort_values('round')
        is_mal = cid == MAL_CLIENT
        color = 'crimson' if is_mal else None
        ax.plot(sub['round'], sub['mean'], marker='o', markersize=4,
                color=color, linewidth=2 if is_mal else 1.3,
                linestyle='--' if is_mal else '-',
                label=f'client_{cid}' + (' (MAL)' if is_mal else ''))
        line_color = ax.lines[-1].get_color()
        ax.fill_between(sub['round'], sub['mean'] - sub['std'], sub['mean'] + sub['std'],
                        color=line_color, alpha=0.18)
    ax.set_xlabel('FL round')
    ax.set_ylabel('Mean |fast − slow|  (per client)')
    ax.set_title('Our temporal advantage: per-round gap trajectory\n'
                 '(the malicious client diverges consistently — SHAP only gives one number)')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = FIG_DIR / 'figure_7_malicious_temporal.png'
    plt.savefig(out, dpi=200, bbox_inches='tight'); plt.close()
    print(f'saved: {out}')


if __name__ == '__main__':
    main()
