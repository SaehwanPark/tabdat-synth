# TabDat-Synth User Manual

This manual documents the current generic package surface. The active package
is `tabdat_synth` under `src/`.

## Public Surface

Main entrypoints:

- `load_generation_spec(path)`
- `generate_from_spec(spec)`
- `fit_synthesizer(spec)`
- `generate_from_fitted(fitted, *, n_samples=None, random_seed=None)`
- `save_fitted_synthesizer(fitted, path)`
- `load_fitted_synthesizer(path)`
- `evaluate_synthetic_data(source_df, synthetic_df, schema, ...)`
- `evaluate_from_spec(spec, synthetic_df, ...)`
- `evaluation_report_to_dict(report)`
- `save_evaluation_report(report, path)`

Main contracts:

- `ColumnSpec`
- `DatasetSchema`
- `GenerationStep`
- `GenerationSpec`
- `ModelBackendSpec`
- `CoefficientTerm`
- `CoefficientSumSpec`
- `BernoulliResponseSpec`
- `BinMapSpec`
- `FittedSynthesizer`
- `MetricResult`
- `EvaluationReport`

The package is currently Python-first. There is not yet a public CLI.

## Runtime Setup

The public package is installable from PyPI with `uv` and requires Python 3.12
or newer:

```bash
uv add tabdat-synth
```

For one-off environment installation:

```bash
uv pip install tabdat-synth
```

For workflows that load local `.env` files through `resolve_va_data_dir(...)`,
install the optional `env` extra:

```bash
uv add "tabdat-synth[env]"
```

The package is still alpha software. Pin to an exact release version in downstream
projects when reproducibility matters.

From the repository root, contributors can run:

```bash
uv run python -c "import tabdat_synth; print(tabdat_synth.__version__)"
```

Included examples:

- DAG example:
  [docs/examples/de_synpuf_beneficiary_phase3.yaml](examples/de_synpuf_beneficiary_phase3.yaml)
- Coefficient-based outcome example:
  [docs/examples/tiny_coefficient_outcomes.yaml](examples/tiny_coefficient_outcomes.yaml)

## Supported Synthesis Styles

The engine currently supports four user-visible synthesis styles.

### 1. Independent Empirical Sampling

Use `kind: empirical` to sample directly from observed source values.

Typical use:

- independent categorical columns
- independent discrete numeric columns
- starting variables in a DAG

### 2. DAG-Conditioned Categorical Synthesis

Use `kind: categorical_model` with `parent_columns`.

Important behavior:

- the direct `parent_columns` define the graph edges
- the DAG is interpreted as a feature-availability graph
- fitting and generation use the full transitive incoming-path closure
- the model therefore sees all already-generated ancestors in stable
  topological order, not only the direct parents
- an optional `feature_columns` tuple overrides that closure and pins the model
  inputs to exactly the listed ancestor columns, in the listed order
- saved categorical-model artifacts expose the resolved `input_columns`

Built-in backends:

- `lightgbm`
- `logistic_regression`

### 3. Deterministic Binning

Use `kind: bin_map` to derive a categorical column from a generated numeric
column through fixed cut points. This step involves no fitting and no random
draw, so it reproduces the same labels for the same input values on every run.

```yaml
- name: agegrp
  kind: bin_map
  output_column: agegrp
  input_column: age_cont
  bin_map:
    breaks: [54, 64, 74, 84]
    labels: ["1", "2", "3", "4", "5"]
```

### 4. Coefficient-Based Outcome Synthesis

This is the generic replacement for the old project-level VA outcome adapter.

Compose three normal steps:

1. `coefficient_sum` builds a linear predictor from an intercept plus typed
   coefficients.
2. `sigmoid_probability` converts that predictor into a probability column.
3. `bernoulli_response` samples a binary response from the predictor using an
   optional intercept and slope adjustment.

This keeps outcome generation inside the same DAG execution model as every
other generated column.

## Source Preparation

Source preparation happens during fitting, after source columns are loaded and
before step artifacts are estimated. The schema controls all coercion:

- categorical columns are cast to strings and then Polars categoricals
- categorical `missing_value_policy: as_category` fills nulls with the shared
  missing category label before fitting
- integer and float columns use non-strict casts, so failed casts become nulls
- numeric `missing_value_policy: mean_impute` fills nulls with the observed
  source-column mean, rounded for integer columns
- `missing_value_policy: forbid` leaves null detection to schema validation

These prepared values are the values used for empirical distributions,
truncated-normal summaries, categorical model targets, and source-side
configured-step fitting.

## Statistical Guarantees

The current engine is a spec-driven simulator, not a privacy-preserving
synthesizer with formal disclosure guarantees. It is designed to preserve the
behaviors declared by each step: empirical marginal sampling, fitted
categorical relationships over resolved ancestor features, truncated-normal
summary behavior, and configured coefficient/probability/response logic.

It is not guaranteed to preserve all high-order associations, downstream model
utility, calibration, or privacy risk. Evaluation reports can help inspect
quality and practical disclosure-risk heuristics, but they do not provide formal
privacy guarantees.

## Evaluation Reports

Use `evaluate_from_spec(...)` when a `GenerationSpec` still points to the source
dataset. The function loads and prepares source columns using the same schema
rules used during fitting, compares only common non-identifier columns, and
records generated-only or otherwise unavailable schema columns as skipped.

```bash
uv run python - <<'PY'
from tabdat_synth import (
    evaluate_from_spec,
    evaluation_report_to_dict,
    generate_from_spec,
    load_generation_spec,
    save_evaluation_report,
)

spec = load_generation_spec("docs/examples/tiny_coefficient_outcomes.yaml")
synthetic = generate_from_spec(spec)
report = evaluate_from_spec(spec, synthetic)

print(evaluation_report_to_dict(report)["evaluated_columns"])
save_evaluation_report(report, "tmp/tiny-evaluation.json")
PY
```

Use `evaluate_synthetic_data(...)` directly when both dataframes are already in
memory:

```python
from tabdat_synth import evaluate_synthetic_data

report = evaluate_synthetic_data(source_df, synthetic_df, spec.schema)
```

Evaluation reports include column-level quality metrics and row-level
disclosure-risk heuristics. They are meant to support review and debugging, not
to certify formal privacy guarantees. The canonical metric guide explains how
columns are selected, how each metric is computed, how to interpret the values,
and how future metrics should fit into the report:
[docs/evaluation-metrics.md](evaluation-metrics.md).

## Step Reference

### `empirical`

Required fields:

- `name`
- `kind: empirical`
- `output_column`
- `source_column` if it differs from `output_column`

### `truncated_normal`

Required fields:

- `name`
- `kind: truncated_normal`
- `output_column`
- `source_column` if it differs from `output_column`

Optional fields:

- `mean_shift`

### `categorical_model`

Required fields:

- `name`
- `kind: categorical_model`
- `output_column`
- `parent_columns`
- `source_column` if it differs from `output_column`

Optional field:

- `feature_columns`

`parent_columns` declares the graph edges. By default the fitted model trains
on the full transitive incoming ancestor closure in stable topological order.
Set `feature_columns` to a non-empty tuple to train and sample on exactly those
columns in that order instead. Every column named in `feature_columns` must
already be an ancestor output, so the tuple can only narrow the closure, never
widen it. The resolved tuple is stored on the fitted artifact as
`input_columns`.

### `bin_map`

Required fields:

- `name`
- `kind: bin_map`
- `output_column`
- `input_column`
- `bin_map`

`bin_map` supports:

- `breaks`: strictly increasing numeric cut points
- `labels`: one string label per resulting bin

Label count must be exactly `len(breaks) + 1`. The step is deterministic: it
applies no random draw and stores no fitted state. Bins follow the Polars `cut`
convention, so a break is the right-inclusive upper edge of the bin below it.
For `breaks: [54, 64, 74, 84]` with `labels: ["1", "2", "3", "4", "5"]`, an
age of `54` maps to `"1"`, `64` to `"2"`, `74` to `"3"`, `84` to `"4"`, and
`85` to `"5"`. The output column must be categorical.

`bin_map` cannot declare `source_column`, `parent_columns`, `feature_columns`,
`response`, or `coefficient_sum`; it depends on the graph only through
`input_column`.

### `coefficient_sum`

Required fields:

- `name`
- `kind: coefficient_sum`
- `output_column`
- `parent_columns`
- `coefficient_sum`

`coefficient_sum` supports:

- `intercept`
- `reference_value`
- `terms`

Each term includes:

- `feature_name`
- `coefficient`
- optional `value`

Use `value` for categorical one-hot style terms. Omit `value` for numeric
passthrough terms.

### `sigmoid_probability`

Required fields:

- `name`
- `kind: sigmoid_probability`
- `output_column`
- `input_column`

### `bernoulli_response`

Required fields:

- `name`
- `kind: bernoulli_response`
- `output_column`
- `input_column`
- `response`

`response` supports:

- `intercept`
- `slope`

`bernoulli_response` outputs are terminal. Later steps cannot depend on them.

## Example Workflow

Run the coefficient-based example in one call:

```bash
uv run python - <<'PY'
from tabdat_synth import generate_from_spec, load_generation_spec

spec = load_generation_spec("docs/examples/tiny_coefficient_outcomes.yaml")
df = generate_from_spec(spec)

print(df.shape)
print(df.select(["eligibility_log_odds", "pEligible", "Eligible"]).head())
PY
```

Fit once, save a bundle, then generate later:

```bash
uv run python - <<'PY'
from tabdat_synth import (
    fit_synthesizer,
    generate_from_fitted,
    load_fitted_synthesizer,
    load_generation_spec,
    save_fitted_synthesizer,
)

spec = load_generation_spec("docs/examples/de_synpuf_beneficiary_phase3.yaml")
fitted = fit_synthesizer(spec)
save_fitted_synthesizer(fitted, "tmp/de_synpuf_bundle")

reloaded = load_fitted_synthesizer("tmp/de_synpuf_bundle")
df = generate_from_fitted(reloaded, n_samples=20, random_seed=999)
print(df.shape)
PY
```

## Model Backends

The backend selector is spec-wide:

```yaml
model_backend:
  kind: logistic_regression
  params:
    solver: liblinear
    max_iter: 500
```

Built-in backend kinds:

- `lightgbm`
- `logistic_regression`

When `model_backend` is omitted, backward-compatible YAML `lgb_n_estimators` and
`lgb_verbosity` fields are still normalized to `ModelBackendSpec(kind="lightgbm", ...)`
for backward compatibility.

If both styles appear, explicit `model_backend` wins.

### Custom Backends

Custom backends are Python-registered only in this phase.

```python
from tabdat_synth.interfaces.models import register_model_backend
```

A custom backend must:

- accept `(X, y, model_backend, random_seed)`
- return a fitted object with `predict_proba(...)`
- set `_sdg_feature_order`

If a saved bundle references a custom backend, that backend must be registered
before `load_fitted_synthesizer(...)`.

## Bundle Format

Saved fitted synthesizers are directory bundles:

- `metadata.json`
- `spec.yaml`
- `steps/`

`metadata.json` records the package and bundle-format versions, creation time,
Python/platform information, selected dependency versions, source format/member,
backend configuration, artifact counts, and stable hashes for the normalized
spec and schema.

The bundle stores enough information to generate without reading the original
source data:

- empirical steps store values and probabilities
- truncated-normal steps store fitted summary statistics and bounds
- categorical-model steps store input columns, learned categories, feature
  order, backend kind, and a pickled fitted model
- coefficient/probability/response steps store normalized step configuration

Pickled model files are trusted-local artifacts. Do not load fitted bundles from
untrusted sources.

Load checks are intentionally strict:

- exact `bundle_format_version` match
- exact package-version match
- required artifact-file presence
- registered backend availability for any categorical-model artifact

## Migration Notes

Older drafts mentioned `va_outcomes` and `VAOutcomeAdapterSpec`. Those are
removed from the main package surface.

Use generic steps instead:

- `coefficient_sum`
- `sigmoid_probability`
- `bernoulli_response`

## Verification

Current verification command:

```bash
uv run --group dev pytest tests/unit tests/regression
```

Current observed result:

```text
61 passed, 2 skipped
```
