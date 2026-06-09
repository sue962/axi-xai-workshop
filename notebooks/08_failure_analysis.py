'''
08 — Failure analysis of the 30% label-flip case + ensemble fix attempt.

Addresses professor feedback (round 2, item 2): "Deepen the 30% failure
mode: Why does it fail? Can you fix it with adaptive thresholding or
ensemble methods?"

Two outputs:
  Part 1 — Why does it fail?
    Per-round per-client consolidation gaps are collected over 3 seeds
    × 2 flip ratios. A boxplot of the per-client gap distribution shows
    that at 100% flip the malicious client (id=0) sits clearly above the
    three honest clients, while at 30% flip its distribution overlaps
    with the honest noise floor — the temporal gap signal is not
    separable from honest cross-round variability at low flip rates.

  Part 2 — Can you fix it?
    A simple ensemble combines the (normalised) consolidation-gap score
    with the (normalised) SHAP-divergence score per client (equal-weight
    sum). We report whether the ensemble recovers separation under the
    30% flip, alongside results under the 100% flip.

Outputs:
    data/processed/failure_per_client.csv   — per-(seed, flip, round, client) raw gaps
    data/processed/failure_summary.csv      — per-(seed, flip) ours/shap/ensemble separations
    data/processed/failure_analysis.txt     — human-readable summary
    figures/figure_8_per_client_gap_distribution.png — Part 1 main figure
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
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent) if '__file__' in dir() else str(Path.cwd()))
from plot_style import set_paper_style
set_paper_style()

NOTEBOOK_DIR = Path.cwd()
PROJECT_ROOT = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == 'notebooks' else NOTEBOOK_DIR

DATA_DIR = PROJECT_ROOT / 'data/processed'
FIG_DIR  = PROJECT_ROOT / 'figures'
OUT_DIR  = DATA_DIR

N_CLIENTS    = 4
HIDDEN       = 64
LOCAL_EPOCHS = 1
BATCH_SIZE   = 256
LR           = 1e-3
ATTACK_TYPES = ['DoS', 'Fuzzy', 'RPM', 'gear']
N_ROUNDS_MAL = 12
SEEDS        = [42, 123, 7]
FLIP_RATIOS  = [1.0, 0.3]
ALPHAS       = [0.3, 0.5, 0.7]   # gap weight in ensemble (1-alpha = SHAP weight)
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

print(f'device = {DEVICE}')
print(f'seeds  = {SEEDS}')
print(f'flips  = {FLIP_RATIOS}')


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


def partition_non_iid(X_train, y_train, src_train, seed, flip_ratio=0.0,
                      malicious_client=0):
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
        yi = y_train[idx].copy()
        if i == malicious_client and flip_ratio > 0:
            flip_mask = rng.random(len(yi)) < flip_ratio
            yi[flip_mask] = 1.0 - yi[flip_mask]
        clients_X.append(X_train[idx])
        clients_y.append(yi)
        clients_src.append(src_train[idx])
    return clients_X, clients_y, clients_src


# -------------------------- malicious detection with per-round logging --------------------------

def run_malicious_detection_detailed(seed, flip_ratio):
    """Returns (per_round_df, ours_scores, shap_scores) for one (seed, flip)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, y_train, src_train, X_test, y_test, src_test = load_and_split(seed)
    clients_X, clients_y, _ = partition_non_iid(
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

    # Per-round per-client gap (averaged over attacks)
    per_round_rows = []
    for cid in range(N_CLIENTS):
        for r in range(1, N_ROUNDS_MAL + 1):
            slow_r = slow_df[slow_df['round'] == r].iloc[0]
            sub = fast_df[(fast_df['client_id'] == cid) & (fast_df['round'] == r)]
            gaps = []
            for _, row in sub.iterrows():
                gaps.append(abs(row['fast_recall'] - slow_r[f'recall_{row["attack"]}']))
            per_round_rows.append({
                'seed': seed, 'flip_ratio': flip_ratio,
                'round': r, 'client_id': cid,
                'gap': float(np.mean(gaps)),
                'is_malicious': int(cid == 0),
            })
    per_round_df = pd.DataFrame(per_round_rows)

    # Mean gap per client (= "ours" score, same as 06)
    ours_scores = {cid: float(per_round_df[per_round_df['client_id'] == cid]['gap'].mean())
                   for cid in range(N_CLIENTS)}

    # SHAP-divergence baseline (same as 06)
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
    shap_scores = {cid: float(np.linalg.norm(v - consensus)) for cid, v in shap_vecs.items()}

    return per_round_df, ours_scores, shap_scores


# -------------------------- ensemble + metrics --------------------------

def minmax_normalise(scores):
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return {k: 0.5 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def separation_and_rank(score, malicious=0):
    honest = {cid: v for cid, v in score.items() if cid != malicious}
    max_honest = max(honest.values())
    sep = score[malicious] / max_honest if max_honest > 0 else float('nan')
    rank1_correct = int(max(score, key=score.get) == malicious)
    return sep, rank1_correct


# -------------------------- main --------------------------

if __name__ == '__main__':
    all_per_round = []
    summary_rows = []

    print('\n' + '=' * 70)
    print('Step 1 — Per-round per-client gap logging (3 seeds × 2 flip_ratios)')
    print('=' * 70)

    for seed in SEEDS:
        for flip in FLIP_RATIOS:
            print(f'\n[seed={seed} flip={flip:.0%}] ...', flush=True)
            per_round_df, ours_scores, shap_scores = run_malicious_detection_detailed(seed, flip)
            all_per_round.append(per_round_df)

            ours_norm = minmax_normalise(ours_scores)
            shap_norm = minmax_normalise(shap_scores)

            ours_sep, ours_r1 = separation_and_rank(ours_scores)
            shap_sep, shap_r1 = separation_and_rank(shap_scores)

            # Weight sweep — gap weight = alpha, SHAP weight = 1 - alpha
            ensemble_results = {}
            for a in ALPHAS:
                ens = {cid: a * ours_norm[cid] + (1 - a) * shap_norm[cid] for cid in ours_scores}
                sep, r1 = separation_and_rank(ens)
                ensemble_results[a] = (sep, r1)

            # Headline ensemble (alpha = 0.5) for back-compat with previous output
            ens_sep, ens_r1 = ensemble_results[0.5]

            row = {
                'seed': seed, 'flip_ratio': flip,
                'ours_sep': ours_sep, 'ours_rank1': ours_r1,
                'shap_sep': shap_sep, 'shap_rank1': shap_r1,
                'ensemble_sep': ens_sep, 'ensemble_rank1': ens_r1,
            }
            for a, (sep, r1) in ensemble_results.items():
                row[f'ens_a{int(a*10):02d}_sep']   = sep
                row[f'ens_a{int(a*10):02d}_rank1'] = r1
            summary_rows.append(row)

            print(f'  ours     sep={ours_sep:.2f}x  rank1={"✓" if ours_r1 else "✗"}')
            print(f'  shap     sep={shap_sep:.2f}x  rank1={"✓" if shap_r1 else "✗"}')
            for a in ALPHAS:
                sep, r1 = ensemble_results[a]
                print(f'  ens α={a:.1f} sep={sep:.2f}x  rank1={"✓" if r1 else "✗"}')

    per_round = pd.concat(all_per_round, ignore_index=True)
    per_round.to_csv(OUT_DIR / 'failure_per_client.csv', index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / 'failure_summary.csv', index=False)

    # -------------------------- Part 1 figure --------------------------
    print('\n' + '=' * 70)
    print('Step 2 — Generating per-client gap distribution figure')
    print('=' * 70)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    flip_titles = {1.0: '100% label flip (extreme)', 0.3: '30% label flip (realistic)'}
    for ax, flip in zip(axes, [1.0, 0.3]):
        sub = per_round[per_round['flip_ratio'] == flip]
        data = [sub[sub['client_id'] == cid]['gap'].values for cid in range(N_CLIENTS)]
        bp = ax.boxplot(
            data, positions=range(N_CLIENTS), widths=0.7, patch_artist=True,
            medianprops={'color': 'black', 'linewidth': 1.8},
            flierprops={'marker': 'o', 'markersize': 5, 'alpha': 0.6},
            whiskerprops={'linewidth': 1.5},
            capprops={'linewidth': 1.5},
            boxprops={'linewidth': 1.5},
        )
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor('#d94545' if i == 0 else '#bbbbbb')
            patch.set_alpha(0.75)
        ax.set_xticks(range(N_CLIENTS))
        ax.set_xticklabels([f'C0\n(malicious)' if cid == 0 else f'C{cid}\n(honest)'
                            for cid in range(N_CLIENTS)])
        ax.set_title(flip_titles[flip])
        ax.set_ylabel('Per-round consolidation gap')
        ax.grid(True, axis='y', alpha=0.3)
    fig.suptitle('Per-client consolidation-gap distribution (3 seeds × 12 rounds)',
                 y=1.02)
    fig.tight_layout()
    out_png = FIG_DIR / 'figure_8_per_client_gap_distribution.png'
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_png}')

    # -------------------------- text summary --------------------------
    lines = []
    lines.append('=' * 70)
    lines.append('30% FAILURE ANALYSIS + ENSEMBLE FIX ATTEMPT')
    lines.append(f'  seeds = {SEEDS}, flip ratios = {FLIP_RATIOS}')
    lines.append('=' * 70)
    lines.append('')

    lines.append('[Part 1] Per-client gap distribution (gap values per round, across seeds)')
    for flip in FLIP_RATIOS:
        lines.append(f'\n  flip_ratio = {flip:.0%}:')
        sub = per_round[per_round['flip_ratio'] == flip]
        for cid in range(N_CLIENTS):
            g = sub[sub['client_id'] == cid]['gap'].values
            tag = 'MALICIOUS' if cid == 0 else 'honest   '
            lines.append(f'    client {cid} ({tag}): '
                         f'mean={g.mean():.4f}  median={np.median(g):.4f}  '
                         f'std={g.std():.4f}  '
                         f'q25={np.quantile(g, 0.25):.4f}  q75={np.quantile(g, 0.75):.4f}')

    lines.append('')
    lines.append('[Part 2] Separation across methods (mean ± std across 3 seeds)')
    for flip in FLIP_RATIOS:
        sub = summary_df[summary_df['flip_ratio'] == flip]
        lines.append(f'\n  flip_ratio = {flip:.0%}:')
        baselines = [
            ('Ours (gap)         ', 'ours_sep', 'ours_rank1'),
            ('SHAP-divergence    ', 'shap_sep', 'shap_rank1'),
        ]
        for method, sep_col, r1_col in baselines:
            lines.append(f'    {method}: '
                         f'sep = {sub[sep_col].mean():.2f}x ± {sub[sep_col].std():.2f}   '
                         f'rank-1 correct: {int(sub[r1_col].sum())}/{len(sub)}')
        for a in ALPHAS:
            tag = f'Ensemble α={a:.1f} (gap+SHAP)'
            sep_col = f'ens_a{int(a*10):02d}_sep'
            r1_col  = f'ens_a{int(a*10):02d}_rank1'
            lines.append(f'    {tag:<19}: '
                         f'sep = {sub[sep_col].mean():.2f}x ± {sub[sep_col].std():.2f}   '
                         f'rank-1 correct: {int(sub[r1_col].sum())}/{len(sub)}')

    # ---- pick best alpha per flip ratio
    lines.append('')
    lines.append('[Part 3] Best alpha per flip ratio (by mean separation)')
    for flip in FLIP_RATIOS:
        sub = summary_df[summary_df['flip_ratio'] == flip]
        best_a, best_sep = None, -np.inf
        for a in ALPHAS:
            sep_col = f'ens_a{int(a*10):02d}_sep'
            m = sub[sep_col].mean()
            if m > best_sep:
                best_a, best_sep = a, m
        r1_col = f'ens_a{int(best_a*10):02d}_rank1'
        lines.append(f'  flip_ratio = {flip:.0%}: best α = {best_a:.1f}   '
                     f'sep = {best_sep:.2f}x   rank-1: {int(sub[r1_col].sum())}/{len(sub)}')

    summary_txt = '\n'.join(lines)
    print('\n' + summary_txt)
    (OUT_DIR / 'failure_analysis.txt').write_text(summary_txt)
    print(f'\nSaved: {OUT_DIR / "failure_analysis.txt"}')
    print(f'Saved: {OUT_DIR / "failure_per_client.csv"}')
    print(f'Saved: {OUT_DIR / "failure_summary.csv"}')
