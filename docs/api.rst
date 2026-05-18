API Reference
=============

.. contents:: Modules
   :local:
   :depth: 1

----

Data
----

Container classes for raw LLM outputs and token-level data returned by the
:mod:`~problm_solver.llama_interface` layer.

.. automodule:: problm_solver.data
   :members:
   :show-inheritance:

----

Model interface
---------------

Thin wrapper around ``llama-cpp-python`` that loads a GGUF model and exposes
generation / probability-query methods used by the rest of the library.

.. automodule:: problm_solver.llama_interface
   :members:
   :show-inheritance:

----

Probability adjustment
----------------------

Adjustment functions and MCMC samplers used by the adjusted-generation
pipeline, including the :class:`~problm_solver.adjust_probs.BranchSampler`
abstract base class and its :class:`~problm_solver.adjust_probs.MetropolisSampler`
implementation.

.. automodule:: problm_solver.adjust_probs
   :members:
   :show-inheritance:

----

Analysis — token probabilities
-------------------------------

Utilities for querying per-token probabilities and sampling from log-prob
dictionaries returned by the model.

.. automodule:: problm_solver.analysis.probabilities
   :members:
   :show-inheritance:

----

CLI
---

Entry point for the interactive command-line interface (``problm-solver``).
Documented here for completeness; most users will interact with the library
programmatically rather than through the CLI module directly.

.. automodule:: problm_solver.cli
   :members:
   :show-inheritance:
