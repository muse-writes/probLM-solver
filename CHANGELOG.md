## v1.3.0 (2026-05-14)

### Feat

- Added candidate token adjustment progress bar for SamplePowerDist
- Silenced Llama output, added tqdm progress and logging.

### Fix

- Context is no longer garbled between adjust_fn calls.
- Fixed decoder treating special characters as strings.
- Power distribution now includes prior tokens.
- **interface**: generate_adjusted now only writes the prompt in 'context'

## v1.2.0 (2026-05-14)

### Feat

- Added candidate token adjustment progress bar for SamplePowerDist
- Silenced Llama output, added tqdm progress and logging.

### Fix

- Context is no longer garbled between adjust_fn calls.
- Fixed decoder treating special characters as strings.
- Power distribution now includes prior tokens.
- **interface**: generate_adjusted now only writes the prompt in 'context'

## v1.2.0 (2026-05-12)

### BREAKING CHANGE

- query_log_probs_next_token() now always returns a
token, type signature of GenerationContext has been updated.

### Feat

- **interface**: Added method for user to change model context.
- **interface**: Finished migration to lower level calls.
- **interface**: query_log_probs_next_token() migrated to eval.
- **llama_interface**: Added helper methods for logprobs and top-k
- generate_adjusted now returns LLMOutputDataFull
- **data**: Added richer data container for LLM output.
- Added interface functions for lower level Llama API.
- Added constants.py (for useful constants)
- SamplePowerDist can now save and reset model state.
- SamplePowerDist now evaluates branch logprob from 1 Llama call.

### Fix

- removed dead tokenizer tests and imports in __init__.py
- Removed dead tokenizer code and exposed eval and reset functions.
- Removed calls to legacy numpy Gumbel distribution.
- correct installation command in README.
- Indexing error in accessing token scores, migrated query_branch
- **docker**: docker-compose run app now functions as intended.
- Tidyup and rollback of context saving features.

## v1.0.1 (2026-04-29)

### Fix

- Metropolis method properly discards equil in SEM calculation.

## v1.0.0 (2026-04-28)

### BREAKING CHANGE

- Function probability adjustment changed to take a
context dataclass as input.

### Feat

- Added function for power sampling using MCMC to cli.py
- Added context dataclass and support for power sampling.
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

- Added more detailed docstrings and an equilibration period for MCMC
- Migrated from legacy rng to np.random.Generator.
- Properly renamed AdjustPower etc. to SampleLowTemp, fixed tests.
- CI for previous commit.
- logging probabilities. logits_all=True only for probs runs.
- Updated Ruff linting rules, adjusted style in code to compensate.
- fixed template and build system. Should now work once Docker is set up correctly.
