# Evaluation Metrics Guide

This guide explains how to read the evaluation report produced by
`evaluate_synthetic_data(...)` and `evaluate_from_spec(...)`. It is written for
collaborators who need to understand the checks well enough to review a
generated dataset, explain a report to another person, or add a future metric
without changing the intent of the current ones.

The short version: the report compares source rows and synthetic rows over the
schema columns that both dataframes share, excluding identifiers. It has two
kinds of metrics:

- quality metrics, which ask whether generated columns resemble source columns
- disclosure-risk heuristics, which ask whether generated rows look too much
  like source rows

The risk metrics are practical review signals. They are not formal privacy
guarantees, not differential privacy, and not a proof that membership inference
or record linkage is impossible.

## Report Lifecycle

There are two public evaluation entrypoints:

- `evaluate_synthetic_data(source_df, synthetic_df, schema, ...)`
- `evaluate_from_spec(spec, synthetic_df, ...)`

Use `evaluate_synthetic_data(...)` when both dataframes are already in memory.
This function evaluates the frames exactly as provided. It does not prepare or
cast source columns before computing metrics.

Use `evaluate_from_spec(...)` when the `GenerationSpec` still points to the
source dataset. This function loads the source dataframe, prepares the source
columns through the schema rules used by the generator, and then delegates to
`evaluate_synthetic_data(...)`.

The report object has this shape:

```python
EvaluationReport(
    source_row_count=...,
    synthetic_row_count=...,
    evaluated_columns=(...),
    skipped_columns=(...),
    quality_metrics=(MetricResult(...), ...),
    disclosure_risk_metrics=(MetricResult(...), ...),
)
```

Each `MetricResult` stores:

- `category`: usually `quality` or `disclosure_risk`
- `name`: stable metric name
- `column`: the column name for column-level metrics, or `None` for row-level
  metrics
- `values`: scalar outputs such as `value`, `mean_delta`, or `p95`
- `details`: context such as which columns were used

`evaluation_report_to_dict(report)` converts the report to plain Python values,
and `save_evaluation_report(report, path)` writes that mapping as JSON.

## Column Selection

The evaluator starts from the schema. For each schema column, it makes one of
three choices:

- evaluate it if it is present in both source and synthetic dataframes and is
  not an identifier
- skip it if it is not an identifier but is missing from either dataframe
- exclude it if the schema marks it as `role: identifier` or the caller passes
  it through `identifier_columns`

This matters because generated-only outcome columns are common. For example, a
spec may synthesize an outcome probability that never existed in the source
file. The evaluator records that column in `skipped_columns` instead of
failing the whole report.

Identifiers are excluded from both quality metrics and row-level risk matching.
That keeps patient IDs, beneficiary IDs, and similar columns from dominating
metrics whose purpose is to compare generated content rather than reproduce
keys.

## Reading The Report

No single metric answers whether a synthetic dataset is good. The useful habit
is to read the report in passes:

1. Check `evaluated_columns` and `skipped_columns`.
   Confirm the report compared the columns you expected. A surprising skipped
   source feature usually means a schema, source, or generation mismatch.

2. Check null-rate deltas first.
   Missingness changes can explain otherwise confusing categorical and numeric
   shifts.

3. Review categorical and numeric quality metrics by column.
   Look for columns whose marginal distribution moved much more than nearby or
   related columns.

4. Review row-level risk heuristics.
   High exact-match or duplicate rates deserve inspection even when marginal
   quality metrics look acceptable.

5. Read related metrics together.
   For example, a low categorical distance plus a high exact-row match rate can
   mean marginal distributions are preserved but records are being copied too
   directly.

The exact thresholds for review are project-specific. A tiny fixture, a heavily
constrained schema, and a large production source file will have different
expected ranges. Treat the numbers as a structured conversation with the data,
not as universal pass/fail labels.

## Shared Metric Rules

Null handling is intentionally explicit:

- every evaluated column gets a separate `null_rate_delta`
- categorical metrics cast values to strings and ignore nulls
- numeric metrics cast values to floats, ignore nulls, and keep finite values
  only

This separation keeps missingness from being counted twice. If missingness
changes, the null metric says so directly; the category and numeric metrics
then describe the non-null values that remain.

The current numeric standard deviation is the population standard deviation
used by NumPy's default `std` behavior. Deltas are always synthetic minus
source, so positive values mean the synthetic statistic is larger.

## Quality Metrics

Quality metrics compare columns independently. They are useful for finding
obvious distribution shifts, invalid generated values, or preparation problems.
They do not measure every relationship between columns.

### `null_rate_delta`

What it checks:

The difference between the synthetic null rate and the source null rate for one
column.

How it is computed:

```text
source_null_rate = source_null_count / source_row_count
synthetic_null_rate = synthetic_null_count / synthetic_row_count
delta = synthetic_null_rate - source_null_rate
```

How to read it:

- `0.0` means missingness has the same rate in source and synthetic data
- positive values mean the synthetic column has more nulls
- negative values mean the synthetic column has fewer nulls

Why it exists:

Missingness can carry signal. If a generator fills nulls unintentionally or
creates too many nulls, downstream models and descriptive statistics may change
even when non-null values look plausible.

Worked example:

| frame | values |
| --- | --- |
| source | `["a", "b", null, "b"]` |
| synthetic | `["a", null, null, "b"]` |

The source null rate is `1 / 4 = 0.25`. The synthetic null rate is
`2 / 4 = 0.50`. The delta is `0.50 - 0.25 = 0.25`.

### `categorical_total_variation_distance`

What it checks:

How far a synthetic categorical distribution moved away from the source
categorical distribution.

How it is computed:

1. Cast non-null source and synthetic values to strings.
2. Compute category proportions in each frame.
3. Take the absolute difference for every category seen in either frame.
4. Sum those differences and divide by two.

```text
TVD = 0.5 * sum(abs(p_synthetic(category) - p_source(category)))
```

How to read it:

- `0.0` means the non-null category proportions match exactly
- `1.0` means the non-null distributions have no overlap
- values in between show the share of probability mass that would need to move
  to make the distributions match

Why it exists:

Categorical columns often represent codes, groups, eligibility flags, or
diagnosis indicators. Total-variation distance gives a bounded and intuitive
way to see whether the generated marginal distribution drifted.

Worked example:

| category | source proportion | synthetic proportion |
| --- | ---: | ---: |
| `a` | 0.50 | 0.25 |
| `b` | 0.50 | 0.25 |
| `c` | 0.00 | 0.50 |

The absolute differences are `0.25`, `0.25`, and `0.50`. Their sum is `1.00`,
so TVD is `0.50`.

### `unseen_synthetic_category_rate`

What it checks:

The share of non-null synthetic categorical values that did not appear in the
source column.

How it is computed:

```text
unseen_rate = synthetic_non_null_values_not_in_source / synthetic_non_null_values
```

How to read it:

- `0.0` means every synthetic category appeared in source data
- higher values mean the generator is emitting novel categories
- `1.0` means every non-null synthetic value is unseen

Why it exists:

Unseen categories can indicate malformed generation, schema drift, or a
legitimate authored category that was absent in a small source sample. The
metric does not decide which case applies, but it makes the issue visible.

Worked example:

Source values are `["a", "a", "b"]`. Synthetic values are
`["a", "c", "c", "b"]`. The unseen synthetic values are the two `c` entries,
so the rate is `2 / 4 = 0.50`.

### `numeric_summary_delta`

What it checks:

Basic numeric marginal summaries for source and synthetic values, plus
synthetic-minus-source deltas.

How it is computed:

The evaluator casts non-null values to floats and keeps finite values. It then
reports:

- `source_mean`, `synthetic_mean`, `mean_delta`
- `source_std`, `synthetic_std`, `std_delta`
- `source_min`, `synthetic_min`, `min_delta`
- `source_max`, `synthetic_max`, `max_delta`

How to read it:

- mean deltas show location shifts
- standard-deviation deltas show spread changes
- min and max deltas show range changes or boundary issues
- `None` means one side had no finite numeric values

Why it exists:

These summaries are simple, but they catch many practical failures quickly:
shifted ages, impossible month counts, collapsed variance, or generated values
outside an expected range.

Worked example:

| frame | values | mean | min | max |
| --- | --- | ---: | ---: | ---: |
| source | `[1, 2, 3]` | 2.0 | 1.0 | 3.0 |
| synthetic | `[2, 4, 6]` | 4.0 | 2.0 | 6.0 |

The mean delta is `4.0 - 2.0 = 2.0`, the min delta is `1.0`, and the max delta
is `3.0`.

### `numeric_ks_statistic`

What it checks:

The largest separation between the empirical cumulative distributions of the
source and synthetic numeric values.

How it is computed:

The evaluator uses `scipy.stats.ks_2samp` on finite numeric values after
non-strict float casting.

How to read it:

- `0.0` means the empirical numeric distributions align exactly
- larger values mean the cumulative distributions separate more
- `None` means one side had no finite numeric values

Why it exists:

Mean, standard deviation, minimum, and maximum can look reasonable even when
the distribution shape changes. The KS statistic is a compact shape check for
numeric columns.

Worked example:

If source values are `[1, 2, 3]` and synthetic values are `[2, 4, 6]`, the two
empirical cumulative distributions are most separated around the lower values.
The current implementation reports `2 / 3` for this example.

## Disclosure-Risk Heuristics

Disclosure-risk metrics look across rows rather than one column at a time.
They are intended to flag suspicious copying, over-concentration, and closeness
to source records. They do not model a specific attacker and do not produce a
formal privacy guarantee.

All row-level risk metrics use the evaluated columns, not skipped columns or
identifier columns.

### `exact_row_match_rate`

What it checks:

The fraction of synthetic rows whose evaluated-column tuple exactly appears in
the source data.

How it is computed:

1. Select the evaluated columns from source and synthetic frames.
2. Convert source rows to a set of row tuples.
3. Count each synthetic row whose tuple appears in that source set.
4. Divide by the synthetic row count.

How to read it:

- `0.0` means no synthetic rows exactly match source rows over evaluated
  columns
- higher values mean more synthetic rows are exact copies over those columns
- repeated matching synthetic rows each count in the numerator

Why it exists:

Exact copying is one of the easiest risks to explain and inspect. A high rate
does not automatically prove a privacy breach, especially for low-dimensional
or highly discrete data, but it is a strong review signal.

Worked example:

| frame | rows over evaluated columns |
| --- | --- |
| source | `(a, 1)`, `(b, 2)` |
| synthetic | `(a, 1)`, `(a, 1)`, `(c, 3)` |

Two of the three synthetic rows exactly appear in the source row set, so the
match rate is `2 / 3`.

### `source_duplicate_row_rate` and `synthetic_duplicate_row_rate`

What they check:

The share of rows that are duplicates beyond the unique-row count, computed
separately for source and synthetic data.

How they are computed:

```text
duplicate_row_rate = (row_count - unique_row_count) / row_count
```

How to read them:

- compare source and synthetic rates together
- a much higher synthetic duplicate rate can signal mode collapse or excessive
  reuse of common records
- a high source duplicate rate can make some synthetic duplication expected

Why they exist:

Synthetic data can look good column by column while repeatedly emitting a small
number of row patterns. Duplicate rates catch that kind of concentration.

Worked example:

Synthetic rows are `(a, 1)`, `(a, 1)`, `(c, 3)`. There are three rows and two
unique rows, so the duplicate rate is `(3 - 2) / 3 = 1 / 3`.

### `nearest_neighbor_distance_summary`

What it checks:

How close each synthetic row is to its nearest source row over numeric
evaluation columns.

How it is computed:

1. Choose numeric evaluated columns by default, or use the caller's
   `nearest_neighbor_columns`.
2. Cast those columns to floats.
3. Drop rows that contain non-finite values in the selected numeric columns.
4. If either side has more than `max_pairwise_rows`, sample rows
   deterministically using `random_seed`.
5. Scale columns using the sampled source mean and standard deviation.
   Columns with zero source standard deviation use scale `1.0`.
6. Compute Euclidean distance from each sampled synthetic row to the nearest
   sampled source row.
7. Report `min`, `p50`, `p95`, `mean`, and `max`, plus the compared row and
   column counts.

How to read it:

- very small distances mean synthetic numeric rows sit close to source numeric
  rows
- a very small `min` is expected in many datasets, especially large ones
- a low `p50` or `p95` is more concerning because it means closeness is common
- no distance values are reported if there are no selected numeric columns or
  no finite rows on either side

Why it exists:

Exact row matching is strict and mostly useful for discrete or rounded data.
Nearest-neighbor distance gives a softer signal for continuous numeric
features, where a generated row may be nearly copied without being exactly
equal.

Worked example:

Source numeric values are `[0, 2]`; the synthetic value is `[1]`. The source
mean is `1`, and the source standard deviation is `1`, so the scaled source
values are `[-1, 1]` and the scaled synthetic value is `[0]`. Its nearest
source distance is `1.0`.

## Metric Combinations Worth Reviewing

Some patterns are more useful than individual numbers:

- High null-rate delta plus large categorical or numeric shifts:
  first inspect missing-value policy and source preparation.
- Low categorical TVD plus high unseen category rate:
  a small amount of novel category mass may be hiding in otherwise stable
  distributions.
- Low numeric mean delta plus high KS statistic:
  the center is stable, but the distribution shape moved.
- Low source duplicate rate plus high synthetic duplicate rate:
  generation may be collapsing to repeated rows.
- Low quality errors plus high exact-row match rate:
  marginal fidelity may be coming from copying source-like rows too directly.
- Low nearest-neighbor `p95`:
  many synthetic numeric rows are close to source numeric rows, not just a few
  isolated cases.

## Limits And Non-Goals

These metrics are intentionally lightweight and transparent. That makes them
easy to run and easy to explain, but it also means they have limits:

- they do not certify privacy
- they do not estimate membership-inference risk under a defined attacker model
- they do not measure all high-order associations
- they do not compare downstream model utility
- they do not decide universal pass/fail thresholds
- they do not replace domain review of sensitive columns

The report is best used as an inspection layer: it highlights where a human or
project-specific policy should look next.

## Adding Future Metrics

Future metrics should preserve the current report contract:

- put pure calculations in `src/tabdat_synth/core/`
- keep dataframe orchestration in `src/tabdat_synth/workflows/`
- return results as `MetricResult` objects
- document the metric name, range, input columns, null handling, and
  interpretation
- add deterministic unit tests with small tables

When adding a metric, be explicit about whether it is a quality metric, a
disclosure-risk heuristic, or something else. The category tells report readers
what question the metric is trying to answer.
