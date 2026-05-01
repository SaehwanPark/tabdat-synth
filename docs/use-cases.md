# Use Cases

This page describes common user groups, practical workflows, and boundaries for TabDat-Synth.

## Research scientists

Researchers may use TabDat-Synth to create simulation datasets that resemble a real cohort or study table.

Useful workflows:

- prototype an analysis when data access is delayed;
- publish runnable examples without restricted source records;
- test estimator behavior under alternative outcome specifications;
- compare methods against controlled synthetic fixtures.

Recommended practice:

- keep the generation specification under version control;
- record random seeds and package versions;
- report synthetic-data limitations alongside results.

## Data scientists and ML engineers

Engineering teams may use TabDat-Synth to build repeatable test datasets for pipelines that expect realistic tabular structure.

Useful workflows:

- create development fixtures with categorical dependencies;
- exercise feature-processing code without production data;
- benchmark model training and scoring paths;
- share reproducible examples in issues, documentation, and demos.

Recommended practice:

- use fitted bundles for generation-only environments;
- include tests that verify expected schema and row counts;
- avoid treating synthetic benchmarks as substitutes for production validation.

## Privacy and governance reviewers

Reviewers may use the package outputs and reports to inspect whether synthetic data appears too close to the source table.

Useful workflows:

- examine exact-row match rates;
- inspect duplicate-row rates;
- review nearest-neighbor distance summaries;
- compare marginal distributions before sharing generated data.

Recommended practice:

- exclude identifiers from evaluation and generation unless a clear simulation need exists;
- interpret metrics as screening signals, not release guarantees;
- combine package reports with institutional disclosure-review procedures.

## Educators and students

Instructors may use TabDat-Synth to distribute datasets that feel realistic without distributing restricted data.

Useful workflows:

- build classroom examples with categorical and numeric variables;
- demonstrate DAG-based data generation;
- teach evaluation of synthetic-data fidelity;
- create assignments that run without private credentials.

Recommended practice:

- keep examples small;
- document which relationships were intentionally encoded;
- avoid implying that synthetic values represent real people.

## Healthcare data collaborators

Healthcare collaborators may use public-style examples to prototype claims-like or cohort-like workflows before working with protected data.

Useful workflows:

- simulate beneficiary-style tables;
- generate binary outcomes from configured coefficients;
- evaluate whether generated records retain expected marginal patterns;
- share non-sensitive fixtures across institutions.

Recommended practice:

- do not publish synthetic data derived from sensitive sources without review;
- document source preparation and missing-value rules;
- avoid reusing original identifiers.
