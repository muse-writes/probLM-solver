## v1.1.0 (2026-04-29)

### Feat

- Added constants.py (for useful constants)
- SamplePowerDist can now save and reset model state.
- SamplePowerDist now evaluates branch logprob from 1 Llama call.

## v1.0.1 (2026-04-29)

### BREAKING CHANGE

- Function probability adjustment changed to take a
context dataclass as input.

### Feat

- Added function for power sampling using MCMC to cli.py
- Added context dataclass and support for power sampling.

### Fix

- Metropolis method properly discards equil in SEM calculation.
- Added more detailed docstrings and an equilibration period for MCMC
- Migrated from legacy rng to np.random.Generator.
- Properly renamed AdjustPower etc. to SampleLowTemp, fixed tests.

## v1.0.0 (2026-04-22)

### Feat

- Implemented framework for adjusted token probabilities
- Can now read and adjust probabilities per-token in a response.
- read and write methods for LLMTokenData.
- Added support for llama.cpp native token+probability output in ModelInstance.query_log_probs()
- Added tokenizer functionality to problm_solver.analysis
- Template directory and file for probability analysis.
- Refactor. data.py now only handles storage.
- Added support for generating output datasets in JSONL file format.
- Added standard Python .gitignore, implemented simple CLI tool.
- Added source files cli.py and llama_interface.py

### Fix

- CI for previous commit.
- logging probabilities. logits_all=True only for probs runs.
- Updated Ruff linting rules, adjusted style in code to compensate.
- fixed template and build system. Should now work once Docker is set up correctly.
