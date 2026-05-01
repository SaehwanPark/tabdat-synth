# Concepts

TabDat-Synth is a spec-driven synthetic data engine for tabular datasets. It favors explicit configuration, small composable steps, and inspectable outputs.

## Motivation

Synthetic tabular data can help when real data is sensitive, restricted, expensive to access, or unsuitable for public examples. Common goals include method development, pipeline testing, teaching, and early collaboration before data-use agreements are complete.

Synthetic data is not automatically private or valid. TabDat-Synth therefore exposes generation assumptions and includes evaluation utilities, rather than treating synthesis as a black box.

## Generation specification

A generation specification describes:

- the source dataset;
- the schema and column roles;
- the step graph;
- model-backend settings;
- random seeds and sample-size choices.

Specifications can be authored in YAML and loaded with `load_generation_spec(...)`.

## DAG execution

Generation steps form a directed acyclic graph. Each step writes one output column. Later steps may depend on earlier outputs.

For `categorical_model` steps, declared `parent_columns` define graph edges. During fitting, the model receives the full incoming ancestor closure, not only direct parents. This gives each conditioned model access to all previously generated context that can reach the target column.

## Supported step kinds

- `empirical`: samples from observed source values.
- `truncated_normal`: samples from a fitted numeric summary with truncation.
- `categorical_model`: fits a categorical target conditional on generated context.
- `coefficient_sum`: creates a linear predictor from configured terms.
- `sigmoid_probability`: maps a linear predictor to a probability.
- `bernoulli_response`: samples a binary response from a probability model.

## Fitted bundles

A fitted bundle stores the learned artifacts needed to generate rows later. It includes metadata, the normalized specification, and step artifacts.

Bundles are useful when source data should stay in a controlled environment. A collaborator can receive a bundle and generate rows without direct source-data access.

Bundle loading is intentionally strict. Package version, bundle-format version, required files, and custom backends are checked before loading.

## Evaluation

Evaluation reports compare source and synthetic data over shared, non-identifier schema columns. They include:

- marginal quality metrics for categorical and numeric columns;
- null-rate deltas;
- exact-row and duplicate-row heuristics;
- nearest-neighbor distance summaries for numeric columns.

These metrics are practical diagnostics. They are not differential privacy, membership-inference protection, or a legal release determination.

## Limitations

TabDat-Synth is alpha software. It is best treated as a transparent research and prototyping tool.

Current limitations include:

- no formal privacy guarantee;
- no public command-line interface;
- limited built-in model backends;
- quality depends strongly on the authored schema and generation graph;
- high-order relationships may not be preserved unless represented in the spec and model choices.
