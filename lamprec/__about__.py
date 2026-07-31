"""Shared constants for the experiment runners."""
SEEDS = list(range(10))          # the paper's ten seeds {0..9} (evaluation only)
TUNING_SEEDS = list(range(100, 110))  # hyperparameter-selection seeds; excluded
                                      # from every reported table (pre-registered
                                      # tuning/eval split, experiments/tuning.py)
ALPHA = 0.05                     # nominal CS level (95% coverage target)

assert not set(SEEDS) & set(TUNING_SEEDS), "tuning seeds must be disjoint from eval seeds"
