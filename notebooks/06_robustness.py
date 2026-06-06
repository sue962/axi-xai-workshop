'''
06 — Robustness extension: multiple seeds + 30% label-flip variant.

Reproduces the headline numbers from notebooks 02/03/05 across three seeds
(42, 123, 7) so we can report mean ± std, and adds a 30% label-flip
malicious-client scenario alongside the existing 100% flip.

Outputs:
    /Users/lucia/Dropbox/USYD/Semester1_2026/AXI/data/processed/robustness_summary.csv
    /Users/lucia/Dropbox/USYD/Semester1_2026/AXI/data/processed/robustness_summary.txt
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
import shap

DATA_DIR = Path('/Users/lucia/Dropbox/USYD/Semester1_2026/AXI/data/processed')
OUT_DIR  = DATA_DIR

# Shared config — matches notebooks 02 / 04 / 05
N_CLIENTS   = 4
HIDDEN      = 64
LOCAL_EPOCHS = 1
BATCH_SIZE  = 256
LR          = 1e-3
ATTACK_TYPES = ['DoS', 'Fuzzy', 'RPM', 'gear']
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Multi-seed sweep
SEEDS = [42, 123, 7]
FLIP_RATIOS = [1.0, 0.3]   # 100% (extreme) and 30% (realistic)

# Sequential CFL (matches notebook 02)
N_ROUNDS_SEQ = 30

# Malicious detection (matches notebook 05)
N_ROUNDS_MAL = 12

print(f'device = {DEVICE}')
print(f'seeds  = {SEEDS}')
print(f'flips  = {FLIP_RATIOS}')


# -------------------------- shared model/utils --------------------------

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


# -------------------------- data prep (per seed) --------------------------

def load_and_split(seed):
    """Load the combined CSV, stratified 80/20 split, standardise."""
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


def partition_non_iid(X_train, y_train, src_train, seed, flip_ratio=0.0,
                      malicious_client=0):
    """Partition 4 non-IID clients; optionally flip labels for malicious client."""
    rng = np.random.default_rng(seed)
    normal_idx = np.where(src_train == 'Normal')[0]
    rng.shuffle(normal_idx)
    normal_chunks = np.array_split(normal_idx, N_CLIENTS)

    clients_X, clients_y, clients_src, clients_role = [], [], [], []
    for i, atk in enumerate(ATTACK_TYPES):
        own = np.where(src_train == atk)[0]
        other_mask = np.isin(src_train, [a for a in ATTACK_TYPES if a != atk])
        other_all = np.where(other_mask)[0]
        other = rng.choice(other_all, size=int(len(other_all) * 0.10), replace=False)
        idx = np.concatenate([normal_chunks[i], own, other])
        rng.shuffle(idx)
        yi = y_train[idx].copy()
        role = 'honest'
        if i == malicious_client and flip_ratio > 0:
            flip_mask = rng.random(len(yi)) < flip_ratio
            yi[flip_mask] = 1.0 - yi[flip_mask]
            role = 'MALICIOUS'
        clients_X.append(X_train[idx])
        clients_y.append(yi)
        clients_src.append(src_train[idx])
        clients_role.append(role)
    return clients_X, clients_y, clients_src, clients_role


# -------------------------- experiment A: sequential CFL --------------------------

def run_sequential_cfl(seed):
    """Returns (gap_ratio, forgetting_dict, transition_gap_mean, steady_gap_mean)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, y_train, src_train, X_test, y_test, src_test = load_and_split(seed)
    clients_X, clients_y, clients_src, _ = partition_non_iid(
        X_train, y_train, src_train, seed, flip_ratio=0.0
    )

    rounds_per_phase = N_ROUNDS_SEQ // len(ATTACK_TYPES)
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
        slow_rows.append({'round': r, 'phase': phase, **sev})

    slow_df = pd.DataFrame(slow_rows)
    fast_df = pd.DataFrame(fast_rows)

    # Forgetting metric: max recall so far minus current
    forgetting = {}
    for atk in ATTACK_TYPES:
        col = slow_df[f'recall_{atk}']
        max_so_far = col.cummax()
        forget_curve = max_so_far - col
        forgetting[atk] = float(forget_curve.iloc[-1])

    # Consolidation gap per round = mean over (client, attack) of |F - S|
    gap_per_round = []
    for r in range(1, N_ROUNDS_SEQ + 1):
        slow_r = slow_df[slow_df['round'] == r].iloc[0]
        fast_r = fast_df[fast_df['round'] == r]
        gaps = []
        for _, row in fast_r.iterrows():
            gap = abs(row['fast_recall'] - slow_r[f'recall_{row["attack"]}'])
            gaps.append(gap)
        gap_per_round.append((r, float(np.mean(gaps))))

    gap_df = pd.DataFrame(gap_per_round, columns=['round', 'gap'])
    transition_rounds = [1, 1 + rounds_per_phase, 1 + 2 * rounds_per_phase, 1 + 3 * rounds_per_phase]
    transition_mask = gap_df['round'].isin(transition_rounds)
    transition_gap_mean = float(gap_df[transition_mask]['gap'].mean())
    steady_gap_mean = float(gap_df[~transition_mask]['gap'].mean())
    gap_ratio = transition_gap_mean / steady_gap_mean if steady_gap_mean > 0 else float('nan')

    return {
        'gap_ratio': gap_ratio,
        'transition_gap_mean': transition_gap_mean,
        'steady_gap_mean': steady_gap_mean,
        **{f'forget_{a}': v for a, v in forgetting.items()},
    }


# -------------------------- experiment B: malicious detection --------------------------

def run_malicious_detection(seed, flip_ratio):
    """Returns (ours_separation, shap_separation, ours_correct, shap_correct)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, y_train, src_train, X_test, y_test, src_test = load_and_split(seed)
    clients_X, clients_y, _, clients_role = partition_non_iid(
        X_train, y_train, src_train, seed, flip_ratio=flip_ratio, malicious_client=0
    )

    global_model = MLP().to(DEVICE)
    slow_rows, fast_rows = [], []
    final_local_models = {}

    for r in range(1, N_ROUNDS_MAL + 1):
        states, weights = [], []
        for i in range(N_CLIENTS):
            local = MLP().to(DEVICE)
            local.load_state_dict(global_model.state_dict())
            sd = train_local(local, clients_X[i], clients_y[i])
            fev = eval_per_source(local, X_test, y_test, src_test)
            for atk in ATTACK_TYPES:
                fast_rows.append({'round': r, 'client_id': i, 'attack': atk,
                                  'fast_recall': fev[f'recall_{atk}']})
            states.append(sd)
            weights.append(len(clients_X[i]))
            if r == N_ROUNDS_MAL:
                final_local_models[i] = {k: v.clone() for k, v in sd.items()}
        global_model.load_state_dict(average_weights(states, weights))
        sev = eval_per_source(global_model, X_test, y_test, src_test)
        slow_rows.append({'round': r, **sev})

    slow_df = pd.DataFrame(slow_rows)
    fast_df = pd.DataFrame(fast_rows)

    # Ours — per-client consolidation gap
    ours_score = {}
    for cid in range(N_CLIENTS):
        sub = fast_df[fast_df['client_id'] == cid]
        gaps = []
        for _, row in sub.iterrows():
            slow_val = slow_df[slow_df['round'] == row['round']].iloc[0][f'recall_{row["attack"]}']
            gaps.append(abs(row['fast_recall'] - slow_val))
        ours_score[cid] = float(np.mean(gaps))

    # SHAP baseline — per-client L2 divergence from consensus
    rng2 = np.random.default_rng(seed)
    bg_idx = rng2.choice(len(X_test), size=40, replace=False)
    ex_idx = rng2.choice(len(X_test), size=60, replace=False)
    X_bg, X_ex = X_test[bg_idx], X_test[ex_idx]

    def shap_vec(sd):
        m = MLP().to(DEVICE)
        m.load_state_dict(sd)
        def f(X): return predict_proba(m, X.astype(np.float32))
        expl = shap.KernelExplainer(f, X_bg)
        sv = np.array(expl.shap_values(X_ex, nsamples=80, silent=True))
        return np.abs(sv).mean(axis=0)

    shap_vecs = {cid: shap_vec(final_local_models[cid]) for cid in range(N_CLIENTS)}
    consensus = np.mean(list(shap_vecs.values()), axis=0)
    shap_score = {cid: float(np.linalg.norm(v - consensus)) for cid, v in shap_vecs.items()}

    # Separation = malicious score / max honest score (only if malicious is the top)
    def sep_ratio(score, malicious=0):
        honest = {cid: v for cid, v in score.items() if cid != malicious}
        max_honest = max(honest.values())
        return score[malicious] / max_honest if max_honest > 0 else float('nan')

    return {
        'ours_separation': sep_ratio(ours_score),
        'shap_separation': sep_ratio(shap_score),
        'ours_mal_score':   ours_score[0],
        'ours_max_honest':  max(v for c, v in ours_score.items() if c != 0),
        'shap_mal_score':   shap_score[0],
        'shap_max_honest':  max(v for c, v in shap_score.items() if c != 0),
        'ours_correct_rank1': int(max(ours_score, key=ours_score.get) == 0),
        'shap_correct_rank1': int(max(shap_score, key=shap_score.get) == 0),
    }


# -------------------------- main sweep --------------------------

if __name__ == '__main__':
    rows_seq = []
    print('\n' + '=' * 70)
    print('PART A — Sequential CFL (gap_ratio, forgetting) × 3 seeds')
    print('=' * 70)
    for seed in SEEDS:
        print(f'\n[seed={seed}] sequential CFL ...', flush=True)
        out = run_sequential_cfl(seed)
        out['seed'] = seed
        rows_seq.append(out)
        print(f'  gap_ratio = {out["gap_ratio"]:.2f}  '
              f'(transition={out["transition_gap_mean"]:.4f}, steady={out["steady_gap_mean"]:.4f})')
        print(f'  forgetting: ' + ', '.join(f'{a}={out[f"forget_{a}"]:+.3f}' for a in ATTACK_TYPES))

    seq_df = pd.DataFrame(rows_seq)
    seq_df.to_csv(OUT_DIR / 'robustness_sequential.csv', index=False)

    rows_mal = []
    print('\n' + '=' * 70)
    print('PART B — Malicious detection (separation) × 3 seeds × 2 flip ratios')
    print('=' * 70)
    for seed in SEEDS:
        for flip in FLIP_RATIOS:
            print(f'\n[seed={seed} flip={flip:.0%}] malicious detection ...', flush=True)
            out = run_malicious_detection(seed, flip)
            out['seed'] = seed
            out['flip_ratio'] = flip
            rows_mal.append(out)
            print(f'  ours sep = {out["ours_separation"]:.2f}x  '
                  f'(mal={out["ours_mal_score"]:.4f}, max_honest={out["ours_max_honest"]:.4f}, '
                  f'rank1={"✓" if out["ours_correct_rank1"] else "✗"})')
            print(f'  shap sep = {out["shap_separation"]:.2f}x  '
                  f'(mal={out["shap_mal_score"]:.4f}, max_honest={out["shap_max_honest"]:.4f}, '
                  f'rank1={"✓" if out["shap_correct_rank1"] else "✗"})')

    mal_df = pd.DataFrame(rows_mal)
    mal_df.to_csv(OUT_DIR / 'robustness_malicious.csv', index=False)

    # ---- Aggregate summary ----
    lines = []
    lines.append('=' * 70)
    lines.append('ROBUSTNESS SUMMARY — mean ± std across 3 seeds (42, 123, 7)')
    lines.append('=' * 70)

    lines.append('\n[A] Sequential CFL (notebook 02 + 03 reproduction):')
    lines.append(f'  Consolidation-gap ratio (transition / steady): '
                 f'{seq_df["gap_ratio"].mean():.2f} ± {seq_df["gap_ratio"].std():.2f}')
    lines.append(f'  Transition-round mean gap: '
                 f'{seq_df["transition_gap_mean"].mean():.4f} ± {seq_df["transition_gap_mean"].std():.4f}')
    lines.append(f'  Steady-round mean gap:      '
                 f'{seq_df["steady_gap_mean"].mean():.4f} ± {seq_df["steady_gap_mean"].std():.4f}')
    lines.append('  Final-round forgetting per attack:')
    for a in ATTACK_TYPES:
        lines.append(f'    {a:5s}: {seq_df[f"forget_{a}"].mean():+.3f} ± {seq_df[f"forget_{a}"].std():.3f}')

    lines.append('\n[B] Malicious-client detection — separation (malicious / max honest):')
    for flip in FLIP_RATIOS:
        sub = mal_df[mal_df['flip_ratio'] == flip]
        lines.append(f'\n  flip_ratio = {flip:.0%}  ({"extreme" if flip >= 1.0 else "realistic"})')
        lines.append(f'    Ours (consolidation gap): '
                     f'{sub["ours_separation"].mean():.2f}x ± {sub["ours_separation"].std():.2f}'
                     f'   (rank-1 correct: {int(sub["ours_correct_rank1"].sum())}/{len(sub)})')
        lines.append(f'    SHAP-divergence baseline: '
                     f'{sub["shap_separation"].mean():.2f}x ± {sub["shap_separation"].std():.2f}'
                     f'   (rank-1 correct: {int(sub["shap_correct_rank1"].sum())}/{len(sub)})')

    summary = '\n'.join(lines)
    print('\n' + summary)
    (OUT_DIR / 'robustness_summary.txt').write_text(summary)
    print(f'\nSaved: {OUT_DIR / "robustness_summary.txt"}')
    print(f'Saved: {OUT_DIR / "robustness_sequential.csv"}')
    print(f'Saved: {OUT_DIR / "robustness_malicious.csv"}')
