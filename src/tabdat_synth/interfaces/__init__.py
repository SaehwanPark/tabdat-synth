"""Boundary adapters for IO and model backends."""

from tabdat_synth.interfaces.bundles import (
  load_fitted_synthesizer,
  save_fitted_synthesizer,
)
from tabdat_synth.interfaces.io import (
  generation_spec_to_dict,
  load_generation_spec,
  load_generation_spec_mapping,
  load_source_dataframe,
)
from tabdat_synth.interfaces.models import (
  list_model_backends,
  register_model_backend,
  unregister_model_backend,
)
from tabdat_synth.interfaces.reports import (
  evaluation_report_to_dict,
  save_evaluation_report,
)

__all__ = [
  "evaluation_report_to_dict",
  "generation_spec_to_dict",
  "list_model_backends",
  "load_fitted_synthesizer",
  "load_generation_spec",
  "load_generation_spec_mapping",
  "load_source_dataframe",
  "register_model_backend",
  "save_evaluation_report",
  "save_fitted_synthesizer",
  "unregister_model_backend",
]
