from __future__ import annotations

from collections.abc import Callable
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegression

from tabdat_synth.domain.contracts import (
  ModelBackendSpec,
)

ModelBackendFactory = Callable[
  [pl.DataFrame, pl.Series, ModelBackendSpec, int],
  Any,
]

_MODEL_BACKEND_REGISTRY: dict[str, ModelBackendFactory] = {}


def register_model_backend(name: str, factory: ModelBackendFactory) -> None:
  """Register a categorical-model backend for Python-side use."""
  backend_name = name.strip()
  if not backend_name:
    raise ValueError("model backend name must be non-empty")
  _MODEL_BACKEND_REGISTRY[backend_name] = factory


def unregister_model_backend(name: str) -> None:
  """Remove a previously registered categorical-model backend."""
  _MODEL_BACKEND_REGISTRY.pop(name, None)


def list_model_backends() -> tuple[str, ...]:
  """Return the available categorical-model backend names."""
  return tuple(sorted(_MODEL_BACKEND_REGISTRY))


def require_model_backend(name: str) -> ModelBackendFactory:
  """Resolve a registered model backend or raise a helpful error."""
  try:
    return _MODEL_BACKEND_REGISTRY[name]
  except KeyError as exc:
    available = ", ".join(list_model_backends()) or "<none>"
    raise ValueError(
      f"model backend '{name}' is not registered; available backends: {available}"
    ) from exc


def fit_categorical_model(
  X: pl.DataFrame,
  y: pl.Series,
  model_backend: ModelBackendSpec,
  random_seed: int,
) -> Any:
  """Fit a registered categorical-model backend."""
  factory = require_model_backend(model_backend.kind)
  model = factory(X, y, model_backend, random_seed)
  setattr(model, "_sdg_backend_kind", model_backend.kind)
  if not hasattr(model, "_sdg_feature_order"):
    raise ValueError(
      f"model backend '{model_backend.kind}' did not set _sdg_feature_order"
    )
  return model


def sample_categorical_feature(
  model: Any,
  X: pl.DataFrame,
  learned_categories: list[str],
  feature_idx: int,
  seed: int,
  feature_order: tuple[str, ...] | None = None,
) -> list[str]:
  """Sample categorical values from predicted class probabilities."""
  resolved_feature_order = feature_order or getattr(model, "_sdg_feature_order", None)
  if resolved_feature_order is None:
    raise ValueError("fitted model is missing encoded feature order metadata")

  X_encoded, _ = _encode_feature_frame(X, resolved_feature_order)
  probs = np.asarray(model.predict_proba(X_encoded), dtype=float)
  feature_rng = np.random.default_rng(seed=seed + feature_idx)

  sampled_indices = [
    int(feature_rng.choice(len(learned_categories), p=prob_row))
    for prob_row in probs
  ]
  return [learned_categories[idx] for idx in sampled_indices]


def _fit_lightgbm_backend(
  X: pl.DataFrame,
  y: pl.Series,
  model_backend: ModelBackendSpec,
  random_seed: int,
) -> lgb.LGBMClassifier:
  params = model_backend.params
  X_encoded, feature_order = _encode_feature_frame(X)
  y_encoded = y.cast(pl.String).to_pandas()
  n_categories = len(y.unique())

  if n_categories == 2:
    model = lgb.LGBMClassifier(
      objective="binary",
      random_state=random_seed,
      verbosity=int(params.get("verbosity", -1)),
      n_estimators=int(params.get("n_estimators", 100)),
    )
  else:
    model = lgb.LGBMClassifier(
      objective="multiclass",
      num_class=n_categories,
      random_state=random_seed,
      verbosity=int(params.get("verbosity", -1)),
      n_estimators=int(params.get("n_estimators", 100)),
    )

  model.fit(X_encoded, y_encoded)
  setattr(model, "_sdg_feature_order", feature_order)
  return model


def _fit_logistic_regression_backend(
  X: pl.DataFrame,
  y: pl.Series,
  model_backend: ModelBackendSpec,
  random_seed: int,
) -> LogisticRegression:
  params = model_backend.params
  X_encoded, feature_order = _encode_feature_frame(X)
  y_encoded = y.cast(pl.String).to_pandas()

  model = LogisticRegression(
    C=float(params.get("C", 1.0)),
    max_iter=int(params.get("max_iter", 500)),
    random_state=random_seed,
    solver=str(params.get("solver", "lbfgs")),
  )
  model.fit(X_encoded, y_encoded)
  setattr(model, "_sdg_feature_order", feature_order)
  return model


def _encode_feature_frame(
  X: pl.DataFrame,
  feature_order: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
  encoded = pd.get_dummies(X.to_pandas(), dtype=float)
  encoded = encoded.astype(float)

  if feature_order is None:
    resolved_order = tuple(str(column) for column in encoded.columns)
    return encoded, resolved_order

  resolved = encoded.reindex(columns=list(feature_order), fill_value=0.0)
  return resolved.astype(float), feature_order


register_model_backend("lightgbm", _fit_lightgbm_backend)
register_model_backend("logistic_regression", _fit_logistic_regression_backend)
