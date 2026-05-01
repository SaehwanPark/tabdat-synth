from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
  """Apply sigmoid after clipping extreme log-odds for numerical stability."""
  return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def logit(p: float | np.ndarray, epsilon: float = 1e-6):
  """Safely compute logit values for probabilities."""
  p_clipped = np.clip(p, epsilon, 1 - epsilon)
  logit_p = np.log(p_clipped / (1 - p_clipped))
  return logit_p


def derive_child_seed(base_seed: int, stream_id: int) -> int:
  """Derive an independent child seed for a specific RNG stream."""
  ss = np.random.SeedSequence([base_seed, stream_id])
  return int(ss.generate_state(1, dtype=np.uint32)[0])
