# TabDat-Synth

![banner](./tabdat-learn-banner.png)

TabDat-Synth is a Python package for spec-driven synthetic tabular data generation. It is intended for research, education, benchmarking, and data-science prototyping where real tabular data is sensitive, unavailable, or inconvenient to share.

The package learns reusable synthesis artifacts from a source table, then generates new rows from an explicit directed acyclic graph (DAG) specification. It supports empirical sampling, conditioned categorical models, numeric summaries, coefficient-based outcomes, fitted bundle reuse, and evaluation reports.

## Motivation

Real tabular data often carries privacy, governance, licensing, or access constraints. These constraints can slow method development, teaching, and reproducible examples. TabDat-Synth provides a small, inspectable synthesis engine that can generate plausible tabular datasets while keeping assumptions visible in configuration.

## Intended users

| User group | Typical use | Relevant docs |
| --- | --- | --- |
| Research scientists | Simulate data for methods work, sensitivity analyses, and reproducible studies. | [Use cases](docs/use-cases.md), [concepts](docs/concepts.md) |
| Data scientists and ML engineers | Prototype pipelines, benchmark models, and create shareable fixtures. | [Getting started](docs/getting-started.md), [user manual](docs/user-manual.md) |
| Privacy and governance reviewers | Inspect whether generated data is too close to source records. | [Evaluation metrics](docs/evaluation-metrics.md), [concepts](docs/concepts.md) |
| Educators and students | Build realistic classroom or workshop datasets without distributing restricted data. | [Getting started](docs/getting-started.md), [examples](docs/examples/) |
| Healthcare data collaborators | Work with public-style examples inspired by claims-data workflows. | [Data-file notes](docs/datafiles-related/) |

## Features

- Spec-driven generation from YAML configuration.
- Directed synthesis workflows with stable topological execution.
- Empirical, truncated-normal, categorical-model, coefficient-sum, sigmoid-probability, and Bernoulli-response steps.
- Built-in categorical backends for LightGBM and logistic regression.
- Reusable fitted bundles for generating later without the original source file.
- Evaluation reports for marginal quality and row-level disclosure-risk heuristics.
- Small public API designed for programmatic Python workflows.

## How it works

A synthesis run has five stages:

1. Load a generation specification and source table.
2. Prepare source columns according to the declared schema.
3. Fit step-level artifacts in DAG order.
4. Sample synthetic rows from the fitted artifacts.
5. Evaluate the synthetic table against source data when appropriate.

The DAG controls which generated columns are available to later steps. For categorical-model steps, declared parents define graph edges, and the fitted model receives the full incoming ancestor context in stable order.

## Installation

TabDat-Synth currently requires Python 3.13 or newer and is distributed from GitHub.

```bash
uv add "tabdat-synth @ git+https://github.com/SaehwanPark/tabdat-synth.git"
```

For one-off installation into an active environment:

```bash
uv pip install "tabdat-synth @ git+https://github.com/SaehwanPark/tabdat-synth.git"
```

For workflows that load local `.env` files through package helpers:

```bash
uv add "tabdat-synth[env] @ git+https://github.com/SaehwanPark/tabdat-synth.git"
```

Because the package is alpha software, pin downstream projects to a commit or tag when reproducibility matters.

## Quick start

Run the coefficient-based example from the repository root:

```python
from tabdat_synth import generate_from_spec, load_generation_spec

spec = load_generation_spec("docs/examples/tiny_coefficient_outcomes.yaml")
df = generate_from_spec(spec)

print(df.shape)
print(df.head())
```

Fit once, save a reusable bundle, and generate later:

```python
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
```

## Documentation

- [Getting started](docs/getting-started.md): installation, first run, and tests.
- [Use cases](docs/use-cases.md): user groups, workflows, and appropriate boundaries.
- [Concepts](docs/concepts.md): motivation, synthesis mechanism, fitted bundles, and limitations.
- [User manual](docs/user-manual.md): public API, step reference, backend configuration, and bundle details.
- [Evaluation metrics](docs/evaluation-metrics.md): fidelity metrics and disclosure-risk heuristics.
- [Synthpop comparison](docs/synthpop-comparison.md): conceptual relationship to R's `synthpop` package.

## Acknowledgement

TabDat-Synth is conceptually inspired by R's `synthpop` package and the broader statistical tradition of synthetic data generation.

> Nowok, B., Raab, G. M., & Dibben, C. (2016). synthpop: Bespoke creation of synthetic data in R. *Journal of statistical software*, 74, 1-26.

Note that this project is a cleanroom Python implementation, not built by reverse engineering `synthpop`, nor reuse `synthpop` source code.
See [docs/synthpop-comparison.md](docs/synthpop-comparison.md) for a full comparison.

## Public API

Common entry points:

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

## Testing

```bash
uv run --group dev pytest tests/unit tests/regression
```

## License

Apache-2.0. See [LICENSE](LICENSE).
