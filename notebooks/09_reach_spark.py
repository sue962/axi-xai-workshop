'''
09 — Reach experiment: generalise the NL consolidation-gap signal to a
second public CAN dataset (HCRL Survival Analysis Dataset, Chevrolet
Spark subset).

Addresses professor feedback (round 2, item 1, marked as "reach"):
"Add one 'reach' experiment: multi-vehicle synthetic data or a different
dataset (even a subset of public CAN datasets) to test generalization."

Pipeline:
  Step 1 — load Chevrolet Spark CAN logs (Flooding / Fuzzy / Malfunction
           + free-driving normal), parse the variable-DLC text format,
           sample to a balanced subset, and write spark_combined.csv.
  Step 2 — run sequential continual federated learning with three
           non-IID clients (one per attack type) and three phases
           (Flooding → Fuzzy → Malfunction) for 30 rounds, using the
           same MLP and FedAvg setup as notebook 02.
  Step 3 — compute per-attack forgetting and the transition-to-steady
           consolidation-gap ratio, and produce a figure mirroring
           Figure 4 on the Spark data.

Outputs:
    data/processed/spark_combined.csv
    data/processed/spark_reach_summary.txt
    figures/fig_9_spark_consolidation_gap.png
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

# -------------------------- paths --------------------------
RAW_DIR  = Path('/Users/lucia/Dropbox/USYD/Semester1_2026/AXI/dataset/Spark')
DATA_DIR = Path('/Users/lucia/Dropbox/USYD/Semester1_2026/AXI/data/processed')
FIG_DIR  = Path('/Users/lucia/Dropbox/USYD/Semester1_2026/AXI/figures')
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------- config (matches nb 02) --------------------------
SEED         = 42
ATTACK_TYPES = ['Flooding', 'Fuzzy', 'Malfunction']    # three, not four
N_CLIENTS    = len(ATTACK_TYPES)                       # one client per attack type
N_ROUNDS     = 30
ROUNDS_PER_PHASE = N_ROUNDS // len(ATTACK_TYPES)       # 10
HIDDEN       = 64
LOCAL_EPOCHS = 1
BATCH_SIZE   = 256
LR           = 1e-3
N_PER_ATTACK = 5000                                    # rows per attack type
N_NORMAL     = 40000                                   # normal rows from FreeDriving

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
torch.manual_seed(SEED); np.random.seed(SEED)


# -------------------------- Step 1: load + preprocess Spark --------------------------

def parse_line(line):
    """Parse one line of the variable-DLC HCRL text format.

    Attack files look like:
        timestamp, CAN_ID, DLC, b0, b1, ..., b{DLC-1}, R|T
    Free-driving files have no trailing flag and are all normal.
    Returns (can_id, dlc, bytes[8], label) or None if malformed.
    """
    parts = [p.strip() for p in line.strip().split(',')]
    if len(parts) < 4:
        return None
    try:
        can_id = int(parts[1], 16)
        dlc    = int(parts[2])
    except ValueError:
        return None
    # last column is flag if it's R/T, else it's a data byte
    flag = None
    body = parts[3:]
    if body and body[-1] in ('R', 'T'):
        flag = body[-1]
        body = body[:-1]
    # Spark FreeDriving file uses space-separated bytes inside one comma cell;
    # attack files use comma-separated bytes. Handle both.
    if len(body) == 1 and ' ' in body[0]:
        body = body[0].split()
    if len(body) < dlc:
        return None
    data = body[:dlc]
    bytes8 = [0] * 8
    for i, b in enumerate(data[:8]):
        try:
            bytes8[i] = int(b, 16)
        except ValueError:
            return None
    label = 1 if flag == 'T' else 0
    return can_id, dlc, bytes8, label


def load_file(path, source_name, max_rows=None):
    """Return DataFrame with columns: can_id, dlc, b0..b7, label, source."""
    rows = []
    with open(path, 'r', encoding='latin-1') as f:
        for line in f:
            r = parse_line(line)
            if r is None:
                continue
            can_id, dlc, b, lbl = r
            rows.append((can_id, dlc, *b, lbl, source_name))
            if max_rows is not None and len(rows) >= max_rows:
                break
    cols = ['can_id', 'dlc'] + [f'b{i}' for i in range(8)] + ['label', 'source']
    return pd.DataFrame(rows, columns=cols)


def build_combined(rng):
    print('\n[Step 1] Loading and combining Spark files ...')
    pieces = []
    for atk in ATTACK_TYPES:
        path = next(RAW_DIR.glob(f'{atk}*_Spark.txt'))
        df = load_file(path, atk)
        attacks_only = df[df['label'] == 1]
        n_take = min(N_PER_ATTACK, len(attacks_only))
        sub = attacks_only.sample(n=n_take, random_state=SEED)
        print(f'  {atk:12s}: total={len(df)}, attack={len(attacks_only)}, sampled={len(sub)}')
        pieces.append(sub)
    # Normal: free-driving file
    free_path = next(RAW_DIR.glob('FreeDrivingData_*_Spark.txt'))
    df_normal = load_file(free_path, 'Normal')
    n_take = min(N_NORMAL, len(df_normal))
    sub_normal = df_normal.sample(n=n_take, random_state=SEED)
    print(f'  {"Normal":12s}: total={len(df_normal)}, sampled={len(sub_normal)}')
    pieces.append(sub_normal)

    combined = pd.concat(pieces, ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    out = DATA_DIR / 'spark_combined.csv'
    combined.to_csv(out, index=False)
    print(f'  Saved: {out} ({len(combined)} rows, attack ratio '
          f'{combined["label"].mean():.3f})')
    return combined


# -------------------------- Step 2: FL utilities (mirrors nb 02) --------------------------

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


# -------------------------- Step 2: split, partition, run CFL --------------------------

def split_and_scale(df):
    feat_cols = ['can_id', 'dlc'] + [f'b{i}' for i in range(8)]
    train_df, test_df = train_test_split(
        df, test_size=0.20, random_state=SEED, stratify=df['source']
    )
    scaler = StandardScaler().fit(train_df[feat_cols].values)
    X_train = scaler.transform(train_df[feat_cols].values).astype(np.float32)
    y_train = train_df['label'].values.astype(np.float32)
    X_test  = scaler.transform(test_df[feat_cols].values).astype(np.float32)
    y_test  = test_df['label'].values.astype(np.float32)
    return X_train, y_train, train_df['source'].values, X_test, y_test, test_df['source'].values


def partition_non_iid(X, y, src, rng):
    """One client per attack type. Each gets its own attack + 1/N of normal +
    10% slice of the other attacks."""
    normal_idx = np.where(src == 'Normal')[0]
    rng.shuffle(normal_idx)
    normal_chunks = np.array_split(normal_idx, N_CLIENTS)
    clients = []
    for i, atk in enumerate(ATTACK_TYPES):
        own = np.where(src == atk)[0]
        other_mask = np.isin(src, [a for a in ATTACK_TYPES if a != atk])
        other_all = np.where(other_mask)[0]
        other = rng.choice(other_all, size=int(len(other_all) * 0.10), replace=False)
        idx = np.concatenate([normal_chunks[i], own, other])
        rng.shuffle(idx)
        clients.append((X[idx], y[idx], src[idx]))
    return clients


def run_sequential_cfl(combined):
    print('\n[Step 2] Sequential CFL on Spark (3 phases × 10 rounds)')
    X_train, y_train, src_train, X_test, y_test, src_test = split_and_scale(combined)
    rng = np.random.default_rng(SEED)
    clients = partition_non_iid(X_train, y_train, src_train, rng)
    for i, (Xi, yi, _) in enumerate(clients):
        print(f'  client {i} ({ATTACK_TYPES[i]:11s}): n={len(Xi)}, attack ratio {yi.mean():.3f}')

    global_model = MLP().to(DEVICE)
    slow_rows, fast_rows = [], []

    for r in range(1, N_ROUNDS + 1):
        phase = min((r - 1) // ROUNDS_PER_PHASE, len(ATTACK_TYPES) - 1)
        active_task = ATTACK_TYPES[phase]
        states, weights = [], []
        for i in range(N_CLIENTS):
            Xi, yi, srci = clients[i]
            mask = (srci == 'Normal') | (srci == active_task)
            Xim, yim = Xi[mask], yi[mask]
            if len(Xim) == 0:
                continue
            local = MLP().to(DEVICE)
            local.load_state_dict(global_model.state_dict())
            sd = train_local(local, Xim, yim)
            fev = eval_per_source(local, X_test, y_test, src_test)
            for atk in ATTACK_TYPES:
                fast_rows.append({'round': r, 'client_id': i, 'attack': atk,
                                  'fast_recall': fev[f'recall_{atk}']})
            states.append(sd); weights.append(len(Xim))
        global_model.load_state_dict(average_weights(states, weights))
        sev = eval_per_source(global_model, X_test, y_test, src_test)
        slow_rows.append({'round': r, 'phase': phase, 'active_task': active_task, **sev})
        if r % 5 == 0 or r == 1:
            print(f'  r={r:2d}  active={active_task:11s}  recall: ' +
                  ', '.join(f'{a}={sev[f"recall_{a}"]:.3f}' for a in ATTACK_TYPES))

    return pd.DataFrame(slow_rows), pd.DataFrame(fast_rows)


# -------------------------- Step 3: metrics + figure --------------------------

def compute_metrics(slow_df, fast_df):
    # forgetting per attack (final round)
    forgetting = {}
    for atk in ATTACK_TYPES:
        col = slow_df[f'recall_{atk}']
        max_so_far = col.cummax()
        forgetting[atk] = float((max_so_far - col).iloc[-1])

    # consolidation gap per round
    gap_per_round = []
    for r in range(1, N_ROUNDS + 1):
        slow_r = slow_df[slow_df['round'] == r].iloc[0]
        fast_r = fast_df[fast_df['round'] == r]
        gaps = []
        for _, row in fast_r.iterrows():
            gap = abs(row['fast_recall'] - slow_r[f'recall_{row["attack"]}'])
            gaps.append(gap)
        gap_per_round.append((r, float(np.mean(gaps))))
    gap_df = pd.DataFrame(gap_per_round, columns=['round', 'gap'])

    transition_rounds = [1 + p * ROUNDS_PER_PHASE for p in range(len(ATTACK_TYPES))]
    t_mask = gap_df['round'].isin(transition_rounds)
    t_mean = float(gap_df[t_mask]['gap'].mean())
    s_mean = float(gap_df[~t_mask]['gap'].mean())
    ratio = t_mean / s_mean if s_mean > 0 else float('nan')

    return forgetting, gap_df, transition_rounds, t_mean, s_mean, ratio


def make_figure(gap_df, transition_rounds, ratio):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(gap_df['round'], gap_df['gap'], marker='o', color='#1f5fa8',
            linewidth=1.4, markersize=4, label='Consolidation gap')
    for tr in transition_rounds:
        ax.axvline(tr, color='#d94545', linestyle='--', alpha=0.55,
                   linewidth=1.0)
    ax.set_xlabel('FL round')
    ax.set_ylabel('Consolidation gap')
    ax.set_title(f'Spark consolidation gap (transition / steady = {ratio:.2f}×)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    fig.tight_layout()
    out = FIG_DIR / 'fig_9_spark_consolidation_gap.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return out


# -------------------------- main --------------------------

if __name__ == '__main__':
    print(f'device = {DEVICE}')
    combined = build_combined(np.random.default_rng(SEED))
    slow_df, fast_df = run_sequential_cfl(combined)
    forgetting, gap_df, transition_rounds, t_mean, s_mean, ratio = compute_metrics(slow_df, fast_df)
    fig_path = make_figure(gap_df, transition_rounds, ratio)

    lines = []
    lines.append('=' * 70)
    lines.append('REACH EXPERIMENT — Chevrolet Spark (HCRL Survival Analysis)')
    lines.append(f'  seed={SEED}, rounds={N_ROUNDS}, phases={ATTACK_TYPES}')
    lines.append('=' * 70)
    lines.append('')
    lines.append(f'Per-attack forgetting (final round):')
    for atk in ATTACK_TYPES:
        lines.append(f'  {atk:12s}: {forgetting[atk]:+.3f}')
    lines.append('')
    lines.append(f'Consolidation gap:')
    lines.append(f'  Transition-round mean: {t_mean:.4f}')
    lines.append(f'  Steady-round mean:     {s_mean:.4f}')
    lines.append(f'  Ratio (transition / steady): {ratio:.2f}x')
    lines.append('')
    lines.append(f'Comparison to HCRL Car-Hacking (2010 Hyundai Sonata):')
    lines.append(f'  HCRL gap ratio: 5.37x (95% bootstrap CI: [3.04, 13.65])')
    lines.append(f'  Spark gap ratio:  {ratio:.2f}x (single-seed, single-vehicle)')

    summary = '\n'.join(lines)
    print('\n' + summary)
    (DATA_DIR / 'spark_reach_summary.txt').write_text(summary)
    print(f'\nSaved: {DATA_DIR / "spark_reach_summary.txt"}')
    print(f'Saved: {fig_path}')
