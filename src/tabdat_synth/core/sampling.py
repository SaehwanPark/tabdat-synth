from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import polars as pl
from scipy.stats import truncnorm

from tabdat_synth.core.probability import logit, sigmoid
from tabdat_synth.core.statistics import compute_category_frequencies
from tabdat_synth.domain.contracts import FreqShift

MissingCategoryHandler = Callable[[str], None]
MISSING_CATEGORY_LABEL = "__MISSING__"
_AGE_GROUP_BREAKS = [54, 64, 74, 84]
_AGE_GROUP_LABELS = ["1", "2", "3", "4", "5"]
_AGE_GROUP_TO_CONTINUOUS = {"1": 57, "2": 67, "3": 77, "4": 87, "5": 92}
_UNKNOWN_AGE_GROUP_FALLBACK = 57


def estimate_empirical_distribution(
  series: pl.Series,
) -> tuple[list[str | int | float], list[float]]:
  """Estimate a discrete distribution from a prepared source series."""
  values = series.to_list()
  if not values:
    raise ValueError(f"cannot sample from empty series '{series.name}'")

  counts: dict[str | int | float, int] = {}
  for value in values:
    if value is None:
      raise ValueError(f"series '{series.name}' contains null values")
    counts[value] = counts.get(value, 0) + 1

  total = float(len(values))
  return list(counts.keys()), [count / total for count in counts.values()]


def sample_empirical_distribution(
  n_samples: int,
  values: list[str | int | float] | tuple[str | int | float, ...],
  probabilities: list[float] | tuple[float, ...],
  seed: int,
) -> np.ndarray:
  """Sample from a persisted discrete empirical distribution."""
  if not values:
    raise ValueError("cannot sample from an empty empirical distribution")
  if len(values) != len(probabilities):
    raise ValueError("empirical distribution values/probabilities length mismatch")

  rng = np.random.default_rng(seed=seed)
  return rng.choice(np.asarray(values, dtype=object), size=n_samples, p=probabilities)


def sample_empirical_values(
  n_samples: int, series: pl.Series, seed: int
) -> np.ndarray:
  """Sample observed values with replacement from a prepared source series."""
  values, probabilities = estimate_empirical_distribution(series)
  return sample_empirical_distribution(n_samples, values, probabilities, seed)


def summarize_truncated_normal_source(
  series: pl.Series,
  minimum: float | None = None,
  maximum: float | None = None,
) -> tuple[float, float, float, float]:
  """Return observed moments and effective bounds for a source series."""
  values = series.cast(pl.Float64).to_numpy()
  if values.size == 0:
    raise ValueError(f"cannot sample from empty series '{series.name}'")

  observed_mean = float(np.mean(values))
  observed_std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
  lower = minimum if minimum is not None else float(np.min(values))
  upper = maximum if maximum is not None else float(np.max(values))

  if lower > upper:
    raise ValueError(f"invalid bounds for '{series.name}': {lower} > {upper}")

  return observed_mean, observed_std, lower, upper


def sample_truncated_normal_from_summary(
  n_samples: int,
  observed_mean: float,
  observed_std: float,
  minimum: float,
  maximum: float,
  seed: int,
  mean_shift: float = 0.0,
  round_to_int: bool = False,
) -> np.ndarray:
  """Sample bounded numeric values from persisted source moments."""
  shifted_mean = observed_mean + mean_shift
  if observed_std == 0.0 or np.isnan(observed_std):
    sampled = np.full(n_samples, np.clip(shifted_mean, minimum, maximum))
  else:
    a = (minimum - shifted_mean) / observed_std
    b = (maximum - shifted_mean) / observed_std
    sampled = truncnorm.rvs(
      a,
      b,
      loc=shifted_mean,
      scale=observed_std,
      size=n_samples,
      random_state=seed,
    )

  if round_to_int:
    sampled = np.rint(sampled).astype(int)

  return sampled


def sample_truncated_normal_values(
  n_samples: int,
  series: pl.Series,
  seed: int,
  minimum: float | None = None,
  maximum: float | None = None,
  mean_shift: float = 0.0,
  round_to_int: bool = False,
) -> np.ndarray:
  """Sample bounded numeric values using observed moments from a source series."""
  observed_mean, observed_std, lower, upper = summarize_truncated_normal_source(
    series,
    minimum=minimum,
    maximum=maximum,
  )
  return sample_truncated_normal_from_summary(
    n_samples,
    observed_mean,
    observed_std,
    lower,
    upper,
    seed,
    mean_shift=mean_shift,
    round_to_int=round_to_int,
  )


def apply_freq_shifts(
  original_freqs: Mapping[str, float],
  shifts: tuple[FreqShift, ...],
  missing_category_handler: MissingCategoryHandler | None = None,
) -> dict[str, float]:
  """Apply category frequency shifts in percentage points."""
  if not shifts:
    return dict(original_freqs)

  shifted_freqs = dict(original_freqs)

  for shift in shifts:
    if shift.source not in shifted_freqs:
      if missing_category_handler:
        missing_category_handler(
          f"source category '{shift.source}' not found in distribution"
        )
      continue
    if shift.dest not in shifted_freqs:
      if missing_category_handler:
        missing_category_handler(
          f"destination category '{shift.dest}' not found in distribution"
        )
      continue

    shift_amount = shift.shift_pct / 100.0
    shifted_freqs[shift.source] -= shift_amount
    shifted_freqs[shift.dest] += shift_amount

  if any(freq < 0 for freq in shifted_freqs.values()):
    negative_cats = [cat for cat, freq in shifted_freqs.items() if freq < 0]
    raise ValueError(f"negative frequencies after shifts: {negative_cats}")

  total = sum(shifted_freqs.values())
  return {cat: freq / total for cat, freq in shifted_freqs.items()}


def generate_continuous_age(
  n_samples: int, original_df: pl.DataFrame, age_drift: float, seed: int
) -> pl.DataFrame:
  """Generate continuous age using a truncated normal distribution with drift."""
  age_stats = original_df.select(
    [
      pl.col("age_cont").mean().alias("mean"),
      pl.col("age_cont").std().alias("std"),
      pl.col("age_cont").min().alias("min"),
      pl.col("age_cont").max().alias("max"),
    ]
  ).row(0)

  age_mean, age_std, age_min, age_max = age_stats
  age_mean += age_drift

  a = (age_min - age_mean) / age_std
  b = (age_max - age_mean) / age_std

  ages = truncnorm.rvs(
    a, b, loc=age_mean, scale=age_std, size=n_samples, random_state=seed
  )

  return pl.DataFrame({"age_cont": ages.round().astype(int)}).with_columns(
    [
      pl.col("age_cont").cast(pl.Int8),
      pl.col("age_cont")
      .cut(breaks=_AGE_GROUP_BREAKS, labels=_AGE_GROUP_LABELS)
      .alias("agegrp"),
    ]
  )


def generate_categorical_start(
  n_samples: int,
  original_df: pl.DataFrame,
  feature_name: str,
  freq_shifts: tuple[FreqShift, ...],
  seed: int,
  missing_category_handler: MissingCategoryHandler | None = None,
) -> pl.DataFrame:
  """Generate a categorical starting feature with optional frequency shifts."""
  rng = np.random.default_rng(seed=seed)

  original_freqs = compute_category_frequencies(original_df[feature_name])
  shifted_freqs = apply_freq_shifts(
    original_freqs, freq_shifts, missing_category_handler
  )

  categories = list(shifted_freqs.keys())
  probabilities = list(shifted_freqs.values())
  sampled_values = rng.choice(categories, size=n_samples, p=probabilities)

  df = pl.DataFrame({feature_name: sampled_values}).with_columns(
    pl.col(feature_name).cast(pl.String).cast(pl.Categorical)
  )

  if feature_name == "agegrp":
    df = df.with_columns(
      pl.col("agegrp")
      .map_elements(
        lambda x: _AGE_GROUP_TO_CONTINUOUS.get(x, _UNKNOWN_AGE_GROUP_FALLBACK),
        return_dtype=pl.Int8,
      )
      .alias("age_cont")
    )

  return df


def generate_responses(
  log_odds: np.ndarray, intercept: float, slope: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
  """Generate risk probabilities and binary outcomes from log-odds."""
  rng = np.random.default_rng(seed=seed)

  risk_probs = sigmoid(log_odds)
  mean_risk_proba = risk_probs.mean()
  mean_risk_log_odds = logit(mean_risk_proba)

  response_log_odds = np.where(
    log_odds >= mean_risk_log_odds,
    intercept + slope * (log_odds - mean_risk_log_odds) + mean_risk_log_odds,
    intercept + log_odds,
  )
  response_probs = sigmoid(response_log_odds)
  outcomes = rng.binomial(n=1, p=response_probs)

  return risk_probs, outcomes
