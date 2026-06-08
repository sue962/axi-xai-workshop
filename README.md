# What It Remembers and Forgets — NL-based Explanation for Federated In-Vehicle Intrusion Detection

Code and experiments for the workshop paper *"What It Remembers and Forgets: Explaining Federated In-Vehicle Intrusion Detection via a Nested-Learning Memory Lens,"* prepared for the KCC 2026 Explainable AI (XAI) Workshop.

## Overview

We apply Google Research's Nested Learning (NL; Behrouz et al., NeurIPS 2025) as an explanation lens for federated learning (FL) based intrusion detection on automotive CAN-bus traffic. Each client's local model is treated as fast memory, and the server-aggregated global model as slow memory. We then quantify what the global model remembers and forgets across rounds and attack types.

Key findings:

- Under sequential-task continual federated learning, the global model retains only the most recently seen attack and silently forgets earlier ones (DoS / RPM / Fuzzy forgetting ≈ +1.000, consistent across seeds).
- Averaged over three seeds (42, 123, 7), the consolidation gap |fast − slow| is 5.37× ± 2.84 larger at task transitions than at steady-state rounds — a temporal signal invisible to SHAP / Shapley.
- Under a 100% label-flip malicious-client scenario, the consolidation-gap signal separates the malicious client 2.29× ± 0.95 more cleanly than SHAP-based detection (1.48× ± 0.05); under a milder 30% flip the gap signal becomes unreliable while SHAP holds up, which we report as a limitation in Section 7 of the paper.

## Data

We use the public HCRL Car-Hacking dataset (Song, Woo, Kim, *Vehicular Communications* vol. 21, 100198, 2020), collected from a real 2010 Hyundai Sonata. Download from <https://ocslab.hksecurity.net/Datasets/car-hacking-dataset> and place under `9) Car-Hacking Dataset/`. The dataset (~850 MB) is not included in this repository.

For the reach experiment we additionally use the Chevrolet Spark subset of the HCRL Survival Analysis Dataset (<https://ocslab.hksecurity.net/Datasets/survival-ids>), placed under `dataset/Spark/`. Also not committed.

## Setup

```bash
conda create -y -n axi python=3.11
conda activate axi
pip install torch pandas numpy scikit-learn shap matplotlib seaborn jupyter ipykernel pymupdf
```

The code uses PyTorch with MPS (Apple Silicon) when available, and falls back to CPU otherwise.

## Notebooks

Run them in order. All runs are deterministic (`SEED=42`).

- `01_preprocess.ipynb` loads the four HCRL attack CSVs and the normal-run log, handles DLC-variable rows, engineers features, and partitions data across four simulated clients (non-IID by attack type).
- `02_fl_baseline.ipynb` runs sequential-task continual federated learning with FedAvg over 30 rounds and logs both the global model (slow memory) and each client's local model (fast memory) per round and per attack type.
- `03_nl_explanation.ipynb` builds the hero remember/forget heatmap and quantifies the 4.1× transition-vs-steady consolidation gap.
- `04_baselines.ipynb` runs SHAP and leave-one-client-out Shapley baselines on an IID FedAvg model, and produces the three-panel comparison figure.
- `05_malicious_detection.ipynb` runs a synthetic label-flipping malicious-client scenario and compares per-client consolidation gap (ours) against per-client SHAP divergence (MDPI-style).
- `06_robustness.py` runs sequential CFL and malicious-client detection across three seeds (42, 123, 7) and two flip ratios (100%, 30%), producing the mean ± std numbers reported in Sections 5–7 of the paper.
- `07_bootstrap_ci.py` re-runs sequential CFL with per-round gap logging and computes a clustered bootstrap (5000 iterations) for the 95% CI on the transition-to-steady gap ratio.
- `08_failure_analysis.py` diagnoses the 30% label-flip failure mode (per-client gap distribution, Figure 8) and sweeps a gap–SHAP ensemble over α ∈ {0.3, 0.5, 0.7}.
- `09_reach_spark.py` runs the reach experiment on the Chevrolet Spark subset of the HCRL Survival Analysis Dataset and produces Figure 9.

To execute end-to-end:

```bash
for nb in 01_preprocess 02_fl_baseline 03_nl_explanation 04_baselines 05_malicious_detection; do
  jupyter nbconvert --to notebook --execute --inplace notebooks/${nb}.ipynb
done
```

After the five notebooks, run the four Python scripts in order: `python notebooks/06_robustness.py`, then `07_bootstrap_ci.py`, `08_failure_analysis.py`, `09_reach_spark.py`.

Intermediate CSVs land in `data/processed/` and PNG figures in `figures/`.

## Status

Work in progress for the KCC 2026 XAI Workshop submission (deadline 9 June 2026). The paper drafts are in `paper/draft_en.md` (English, submission language) and `paper/draft_ko.md` (Korean companion).

## Contact

Suebin Lee — Master's student, University of Sydney. Volunteer, Dual Lab.
