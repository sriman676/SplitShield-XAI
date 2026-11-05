"""Self-improvement: search -> distil/prune/quantise -> gate -> promote.

Improves the MODEL each run, never the pipeline source (guarded Layer C only
emits isolated, test-gated files under ``models/generated/``). Reproducible:
every candidate is built from a seeded Config.
"""
