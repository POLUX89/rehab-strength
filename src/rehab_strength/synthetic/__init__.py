"""Synthetic data generation for the app's demo mode.

Manual version (no SDV): plausible marginals + an AR(1) latent state for the
temporal structure + derived columns recomputed with the real pipeline. The
parameters are generic (physiologically plausible), NOT fitted to the user's
real data: not even summary statistics of health data touch the repo. The
generated CSVs live in data/synthetic/ and ARE versioned.
"""
