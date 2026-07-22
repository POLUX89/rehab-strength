# Synthetic demo data

These CSVs are **100% synthetic** — generated with generic, physiologically
plausible parameters. They were **not** fitted to the author's real data: no
real health values, and not even summary statistics of them, exist here.

They power the app's **"🧪 Try with synthetic data"** demo button, and unlike
everything else under `data/`, they are **versioned on purpose** so anyone can
explore the deployed app or reproduce the analyses.

Regenerate deterministically (seeded) with:

```bash
make synthetic          # = python -m rehab_strength.synthetic.generate
```

How they're built (see `src/rehab_strength/synthetic/generate.py`): a daily
AR(1) latent "wellness" state induces temporal autocorrelation and the
cross-column correlations (more stress → worse score, more sleep → better
recovery); derived columns (Efficiency, z-scores, sigmoid score, nap scoring)
are recomputed with the ingestion pipeline's own formulas; workouts come from a
rule-based training simulator (weekly split, linear progression, deloads).
