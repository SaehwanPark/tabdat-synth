# Getting Started

This guide is for users who want to install TabDat-Synth, run a public example, and verify that the repository works locally.

## Requirements

- Python 3.13 or newer.
- `uv` for the documented commands.
- A POSIX-like shell for the inline examples.

## Install from GitHub

Add the package to a project:

```bash
uv add "tabdat-synth @ git+https://github.com/SaehwanPark/tabdat-synth.git"
```

Install into the active environment:

```bash
uv pip install "tabdat-synth @ git+https://github.com/SaehwanPark/tabdat-synth.git"
```

Install the optional environment-file extra only when a workflow needs local `.env` loading:

```bash
uv add "tabdat-synth[env] @ git+https://github.com/SaehwanPark/tabdat-synth.git"
```

## Run a minimal example

From the repository root:

```bash
uv run python - <<'PY'
from tabdat_synth import generate_from_spec, load_generation_spec

spec = load_generation_spec("docs/examples/tiny_coefficient_outcomes.yaml")
synthetic = generate_from_spec(spec)

print(synthetic.shape)
print(synthetic.head())
PY
```

This example loads a YAML specification, fits the necessary artifacts, and returns a Polars dataframe.

## Save and reload a fitted bundle

Use a fitted bundle when source data should not be required at generation time.

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
synthetic = generate_from_fitted(reloaded, n_samples=20, random_seed=999)
print(synthetic.shape)
PY
```

Do not load bundle artifacts from untrusted sources. Categorical-model artifacts may contain pickled model files.

## Evaluate generated data

```bash
uv run python - <<'PY'
from tabdat_synth import (
  evaluate_from_spec,
  evaluation_report_to_dict,
  generate_from_spec,
  load_generation_spec,
)

spec = load_generation_spec("docs/examples/tiny_coefficient_outcomes.yaml")
synthetic = generate_from_spec(spec)
report = evaluate_from_spec(spec, synthetic)

print(evaluation_report_to_dict(report)["evaluated_columns"])
PY
```

Evaluation reports are review aids. They do not certify privacy.

## Run tests

```bash
uv run --group dev pytest tests/unit tests/regression
```
