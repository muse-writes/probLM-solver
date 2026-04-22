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
