# probLM-solver Codebase Assessment

## Project Overview
**probLM-solver** is a Python library built on top of `llama.cpp` that generates and performs statistical analysis on language model outputs.

- **Repository**: https://github.com/muse-writes/probLM-solver
- **Author**: Clio (d.k.johnson@lancaster.ac.uk)
- **Python Version**: >=3.13, <4.0
- **Build System**: uv_build
- **Status**: Version 0.0.0 (early development)

---

## Directory Structure
```
probLM-solver/
├── src/problm_solver/          # Main source code (installed as `problm_solver`)
│   ├── __init__.py              # Package initialisation (empty docstring)
│   ├── cli.py                   # Command line interface
│   ├── llama_interface.py       # llama.cpp wrapper interface
│   ├── data.py                  # LLM output data containers and serialisation
│   └── analysis/                # Statistical analysis subpackage
│       ├── __init__.py          # Re-exports public analysis API
│       ├── tokenizer.py         # Tokenizer ABC and implementations
│       └── probabilities.py     # Sampling utility + positional probability stub
├── tests/
│   ├── __init__.py
│   ├── test_import.py           # Basic import test
│   ├── test_data.py             # Tests for data.py
│   ├── test_cli.py              # Tests for cli.py
│   ├── test_llama_interface.py  # Tests for llama_interface.py
│   └── test_probabilities.py   # Tests for analysis/probabilities.py
├── docs/
│   └── index.md                 # Documentation (minimal)
├── pyproject.toml               # Project configuration & dependencies
├── Dockerfile                   # Multi-stage Docker build (dev + app targets)
├── docker-compose.yml           # Compose configuration
├── mkdocs.yml                   # Documentation build config
├── README.md                    # Project README
└── .gitignore                   # Standard Python gitignore
```

---

## Core Modules

### 1. **data.py**
**Purpose**: Standalone data containers for LLM output; handles serialisation and deserialisation.

**Key Classes**:
- `LLMOutputData`: Stores a prompt and its associated LLM responses
  - `__init__(prompt: str, data: npt.NDArray[Any])`: Constructed with raw data, no dependency on `ModelInstance`
  - `write(fname: str)`: Saves to JSONL format; each line has `id`, `prompt`, `response` fields; sets `self.written = True`
  - `read(fname: str)`: Loads from a JSONL file; populates `self.prompt`, `self.data`, and sets `self.written = True`
  - `self.written`: Tracks whether in-memory data is in sync with disk

- `LLMTokenData`: Stores a single tokenized LLM response paired with per-token probabilities
  - `__init__(prompt: str, tokens: list[str], probs: list[float])`: Validates that `len(tokens) == len(probs)`, raising `TokenProbError` otherwise
  - `write(fname: str)`: Saves to JSON format as a single record with `prompt`, `tokens`, `probs` fields; sets `self.written = True`
  - `read(fname: str)`: Loads from a JSON file; populates `self.prompt`, `self.tokens`, `self.probs`; sets `self.written = True`
  - `self.tokens`: Ordered list of token strings
  - `self.probs`: Per-token probabilities (same length as `tokens`); `probs[i]` is the probability of `tokens[i]` at its position

- `TokenProbError(ValueError)`: Raised by `LLMTokenData.__init__` when `len(tokens) != len(probs)`

- `LLMNextTokenData`: Stores the top-M most likely next tokens and their log-probabilities given a prompt and current output vector
  - `__init__(prompt: str, output_vec: list[int], top_m_tokens: dict[str, float])`: `output_vec` is the current token ID sequence; `top_m_tokens` maps token strings to log-probs
  - `write(fname: str)`: Saves to JSON format with `prompt`, `output_vec`, `top_m_tokens` fields; sets `self.written = True`
  - `read(fname: str)`: Loads from JSON; populates all three fields; sets `self.written = True`
  - `self.written`: Tracks whether in-memory data is in sync with disk

**Dependencies**: `json`, `numpy`

---

### 2. **llama_interface.py**
**Purpose**: Wrapper around `llama_cpp` for local model inference. Owns the responsibility of generating all data container types.

**Module-level**:
- `AdjustFn = Callable[[dict[str, float]], dict[str, float]]`: Type alias for a probability adjustment callable. Receives a top-M `{token: log_prob}` dict and returns a modified log-prob dict; values need not be normalised.

**Key Classes**:
- `ModelInstance`: Represents a loaded GGUF model
  - `__init__(fname: str, context: str, logits_all: bool = False)`: Loads model via `Llama`; hard-coded `n_ctx=2048`; `logits_all` forwarded to `Llama` — set `True` for logprob runs
  - `query() -> str`: Single inference call; hard-coded `max_tokens=512`; uses chat completion with `role='user'`
  - `query_n_times(n: int) -> npt.NDArray[Any]`: Calls `query()` N times, returns numpy array
  - `generate_data(n_samples: int) -> LLMOutputData`: Calls `query_n_times()` and wraps result in `LLMOutputData`
  - `query_log_probs() -> LLMTokenData`: Single inference call with `logprobs=True, top_logprobs=1`; extracts token strings and probabilities (`exp(logprob)`) directly from the API response; returns `LLMTokenData`
  - `query_log_probs_next_token(context_tokens: list[int], n_tokens: int) -> LLMNextTokenData`: Calls `create_completion()` with `max_tokens=1` and `top_logprobs=n_tokens`; `context_tokens` is a list of integer token IDs passed directly as the prompt; returns `LLMNextTokenData` with the top-M `{token_str: log_prob}` dict
  - `get_tokenizer() -> LlamaTokenizer`: Returns a `LlamaTokenizer` backed by this model's `Llama` instance
  - `_format_chat_prompt() -> list[int]`: Private method; applies the model's chat template to `self.context` via `self._llm._chat_handler` (the same formatter used internally by `create_chat_completion()`), then tokenises the result with `add_bos=False, special=True`; returns a list of integer token IDs as the initial context for `generate_adjusted()`
  - `generate_adjusted(n_tokens: int, adjust_fn: AdjustFn, max_tokens: int) -> LLMOutputData`: Token-by-token generation loop; at each step calls `query_log_probs_next_token`, passes `top_m_tokens` to `adjust_fn`, samples the next token via `sample_from_logprobs`, converts it to a token ID and appends to context; breaks on EOS or empty token ID list; decodes the generated IDs and returns wrapped in `LLMOutputData`

**Dependencies**: `math`, `collections.abc.Callable`, `llama_cpp`, `numpy`, `problm_solver.data`, `problm_solver.analysis.tokenizer`, `problm_solver.analysis.probabilities`

---

### 3. **cli.py**
**Purpose**: Interactive command-line interface for the full generate-and-save workflow.

**Constants**:
- `PROBLM_DIR`: `~/.problm-solver/`
- `MODELS_DIR`: `~/.problm-solver/models/`
- `RESPONSES_DIR`: `~/.problm-solver/datasets/responses/`
- `PROBS_DIR`: `~/.problm-solver/datasets/probabilities/`
- `NUMBER_OF_FUNCTIONS = 2`, `GEN_DATA = 1`, `PROBS = 2`: Function selection constants

**Functions**:
- `ensure_models_dir() / ensure_responses_dir() / ensure_probs_dir()`: Create directories if absent; each returns its path
- `list_models() -> list[Path]`: Sorted list of `.gguf` files in `MODELS_DIR`
- `get_responses_path(model_path: Path) -> Path`: Timestamped `.jsonl` path inside `RESPONSES_DIR`
- `get_probs_path(model_path: Path) -> Path`: Timestamped `prob_*.json` path inside `PROBS_DIR`
- `ui_select_model() -> Path`: Interactive model picker
- `ui_select_function() -> int`: Interactive function picker; returns `GEN_DATA` (1) or `PROBS` (2)
- `ui_gen_data(model, model_path)`: Prompts for sample count, calls `model.generate_data()`, delegates to `ui_save_data`
- `ui_save_data(fname: str, data: LLMOutputData)`: Prompts user to confirm saving; calls `data.write()`
- `ui_save_token_data(fname: str, data: LLMTokenData)`: Prompts user to confirm saving; calls `data.write()`
- `ui_get_probs(model, model_path)`: Calls `model.query_log_probs()`, delegates to `ui_save_token_data`
- `main()`: Full workflow — select model → enter prompt → select function → load model (with `logits_all=True` iff probs run) → dispatch to `ui_gen_data` or `ui_get_probs`

**Dependencies**: `problm_solver.llama_interface`, `problm_solver.data`

---

### 4. **analysis/\_\_init\_\_.py**
Re-exports the public API of the `analysis` subpackage:
`Tokenizer`, `WordTokenizer`, `LlamaTokenizer`, `Token`, `TokenSequence`, `sample_from_logprobs`

---

### 5. **analysis/tokenizer.py**
**Purpose**: Tokenizer abstraction and implementations for splitting LLM response strings into token sequences for statistical analysis.

**Type Aliases**:
- `Token = str`: A single decoded token string
- `TokenSequence = list[Token]`: An ordered list of tokens from one response

**Key Classes**:
- `Tokenizer` (ABC): Abstract base class; defines `tokenize(text: str) -> TokenSequence`
- `WordTokenizer(Tokenizer)`: Regex-based tokenizer; no external dependencies
  - Splits on word boundaries; keeps contractions (e.g. `"it's"`) as single tokens; punctuation becomes individual tokens; whitespace is discarded
  - `__init__(*, lowercase: bool = False)`
- `LlamaTokenizer(Tokenizer)`: Tokenizer backed by a loaded `llama_cpp.Llama` instance
  - Uses the model's own BPE vocabulary; each token ID is decoded individually via `detokenize([id])` to preserve token boundaries
  - `__init__(llama: _LlamaInterface)`
- `_LlamaInterface` (Protocol, private): Structural protocol describing the `tokenize` and `detokenize` methods used from `llama_cpp.Llama`; avoids a hard module-level import of `llama_cpp`

**Dependencies**: `re`, `abc`, `typing`

---

### 6. **analysis/probabilities.py**
**Purpose**: Token sampling utility and positional probability analysis stub.

**Functions**:
- `sample_from_logprobs(log_probs: dict[str, float]) -> str`: Samples a token string from a log-probability distribution. Shifts by max before `exp()` for numerical stability, renormalises, and samples via `np.random.choice`. Input values need not correspond to a normalised distribution.

**Key Classes**:
- `Probabilities`: Stub implementation; not yet functional
  - `__init__(data: LLMOutputData, entry: str)`
  - `evaluate() -> None`: Skeleton only; body is `pass`

**Active linter errors in `Probabilities`** (see Known Issues):
- `F821`: `List` used but not imported; should be built-in `list[str]`
- `F821`: `data` referenced in `evaluate()` but not in scope; should be `self.data`
- `F841`: local variable `probabilities` assigned but never used
- `B007`: loop variables `ii` and `data_entry` unused (loop body is `pass`)

**Dependencies**: `numpy`, `problm_solver.data`

---

## Architecture

### Dependency Direction
```
cli.py → ModelInstance (llama_interface.py) → LLMOutputData (data.py)
                                             → LLMTokenData (data.py)
                                             → LLMNextTokenData (data.py)
                                             → LlamaTokenizer (analysis/tokenizer.py)
                                             → sample_from_logprobs (analysis/probabilities.py)
cli.py → LLMOutputData (data.py)
analysis/probabilities.py → LLMOutputData (data.py)
analysis/tokenizer.py → (no intra-package deps)
data.py → (no intra-package deps)
```

All data container classes (`LLMOutputData`, `LLMTokenData`, `LLMNextTokenData`) have no dependency on `ModelInstance`. `ModelInstance` constructs and returns them via its various generation methods.

### Data Flow
1. User selects a GGUF model file from `~/.problm-solver/models/`
2. User enters a prompt, then selects a function (gen_data or probs)
3. `ModelInstance` is created with the model path, prompt, and `logits_all=True` if a probs run was selected
4. **Text responses**: `model.generate_data(n)` queries the LLM N times → `LLMOutputData`; saved to `~/.problm-solver/datasets/responses/`
5. **Token + probability responses**: `model.query_log_probs()` queries once with `logprobs=True` → `LLMTokenData`; saved to `~/.problm-solver/datasets/probabilities/`
6. **Adjusted generation**: `model.generate_adjusted(n_tokens, adjust_fn, max_tokens)` drives a token-by-token loop; at each step `adjust_fn` receives the top-M log-prob dict and returns a modified one; the next token is sampled from the adjusted distribution and appended to the context as a token ID → `LLMOutputData`
7. Token sequences for statistical analysis are obtained by calling `model.get_tokenizer()` and applying `tokenizer.tokenize()` to each response in an `LLMOutputData`

---

## Dependencies

### Runtime (`dependencies` in pyproject.toml)
- `llama-cpp-python`: LLM inference via llama.cpp
- `numpy`: Array operations
- `poethepoet`: Task runner — kept as a runtime dependency so it is available inside the Docker app container (installed with `--no-dev`)

### Development (`dependency-groups.dev`)
- `pytest >= 8.3.4` + `pytest-mock`, `pytest-xdist`: Testing
- `coverage[toml] >= 7.6.10`: Code coverage (minimum 50%)
- `ruff >= 0.9.2`: Linting & formatting
- `pre-commit >= 4.0.1`: Git hooks
- `commitizen >= 4.3.0`: Semantic versioning
- `mkdocs-material >= 9.5.21`: Documentation
- `typeguard >= 4.4.1`, `ty >= 0.0.6`: Type checking
- `ipython`, `ipykernel`, `ipywidgets`: Jupyter support
- `codespell >= 2.4.1`: Spell checking

---

## Docker Setup

### Dockerfile (multi-stage)
- **`dev` stage**: Based on `ghcr.io/astral-sh/uv:python3.13-trixie`; sets up venv at `/opt/venv`; intended for VS Code Dev Containers
- **`app` stage**: Based on `python:3.13-slim`; runs `uv sync --no-dev --no-editable --frozen`; venv at `/workspaces/problm-solver/.venv`; ENTRYPOINT is `.venv/bin/poe`, CMD is `serve`

### docker-compose.yml
- **`devcontainer`**: Runs by default (`docker compose up`); mounts `..:/workspaces`
- **`app`**: Requires explicit profile (`docker compose up app`); has `stdin_open: true` and `tty: true` for interactive CLI; mounts `~/.problm-solver:/home/user/.problm-solver` for model and dataset persistence

### Running the App
```sh
# From host
docker compose up app

# Equivalent from inside Dev Container
poe serve
```

---

## Poe Tasks (`[tool.poe.tasks]`)

The poe executor is set to `type = "virtualenv"` so tasks resolve executables from the project venv rather than the system PATH.

```bash
poe serve           # Run the CLI (calls the `problm-solver` entry point)
poe lint            # Run pre-commit hooks on all files
poe test            # Run tests + coverage report + coverage XML
poe docs            # Build documentation
poe docs --serve    # Serve documentation locally with live reload
```

### Entry Point
`[project.scripts]` declares:
```
problm-solver = "problm_solver.cli:main"
```
This is installed into the venv's `bin/` by `uv sync` and is the target of `poe serve`.

---

## Code Quality & Configuration

### Testing
- **Framework**: pytest with doctest modules
- **Coverage**: Minimum 50% (`fail_under = 50`)
- **Options**: Exit on first failure, verbose (v=2), JUnit XML reports
- **Paths**: `src/`, `tests/`
- **Test files**: `test_import.py`, `test_data.py`, `test_cli.py`, `test_llama_interface.py`, `test_probabilities.py`
- **Doctests**: `WordTokenizer` and `LlamaTokenizer` have inline doctests collected by `--doctest-modules`
- **Mocking**:
  - `llama_cpp.Llama` is patched at construction time in all `ModelInstance` tests; the resulting `MagicMock` is configured per-test via `return_value` overrides
  - `_format_chat_prompt` and `query_log_probs_next_token` are patched with `patch.object` in `generate_adjusted` tests; `sample_from_logprobs` is patched at the `llama_interface` module level
  - `contextlib.ExitStack` is used in the `gen_adj_model` fixture to manage multiple simultaneous patches cleanly
  - `LlamaTokenizer` is tested via `_LlamaInterface`-compatible mocks
  - CLI filesystem interactions use `monkeypatch` against `tmp_path`

### Linting (Ruff)
- **Line Length**: 100 characters
- **Target Python**: 3.13
- **Docstring Convention**: reStructuredText / PEP 257 (`:param:`, `:returns:` directives)
- **Quotes**: Single (inline and multiline)
- **Imports**: Absolute only (`ban-relative-imports = "all"`)

### Versioning
- Conventional Commits enforced via Commitizen
- `cz bump` → updates `CHANGELOG.md`, bumps version, creates git tag

---

## Known Issues

### Active
1. **Unreachable input validation loop** (`cli.py`, `ui_gen_data()`):
   - `data_size = int(input(...))` either succeeds (always an `int`) or raises an exception
   - The `while not isinstance(data_size, int)` guard below it can never be reached

2. **Hard-coded model parameters** (`llama_interface.py`):
   - `n_ctx=2048` and `max_tokens=512` are not configurable

3. **`Probabilities` class is a non-functional stub** with multiple linter errors (`probabilities.py`):
   - `F821`: `List` used but not imported; should be built-in `list[str]`
   - `F821`: `data` referenced in `evaluate()` but not in scope; should be `self.data`
   - `F841`: local variable `probabilities` assigned but never used
   - `B007`: loop variables `ii` and `data_entry` unused (loop body is `pass`)

4. **`FBT001` on `_LlamaInterface` protocol** (`analysis/tokenizer.py`):
   - `add_bos: bool` and `special: bool` are flagged as boolean positional arguments
   - These mirror the external `llama_cpp.Llama` API signature and cannot be made keyword-only

5. **`generate_adjusted()` returns `LLMOutputData`** — richer return type is a known limitation:
   - **TODO**: Replace with a container that records the full sequence of `LLMNextTokenData` snapshots (one per generated token), giving access to the adjusted conditional distributions at each step for post-hoc analysis

### Resolved
- ✓ `LLMOutputData.read()` implemented
- ✓ `LLMOutputData` dependency on `ModelInstance` removed; `generate_data()` added to `ModelInstance`
- ✓ `llama-cpp-python` and `numpy` added to runtime dependencies
- ✓ `poethepoet` moved to runtime dependencies for Docker app container compatibility
- ✓ Poe executor changed from `simple` to `virtualenv`
- ✓ `poe serve` wired up to `problm-solver` entry point (was a placeholder echo)
- ✓ Intra-package imports converted from bare names to absolute (`problm_solver.*`)
- ✓ Docker `app` service configured with `stdin_open`, `tty`, and volume mount for model/data persistence
- ✓ Test file bug fixed (`import problm-solver` → `import problm_solver`)
- ✓ Test suite created: `test_data.py`, `test_cli.py`, `test_llama_interface.py`
- ✓ `analysis/` subpackage created with `Tokenizer` ABC, `WordTokenizer`, and `LlamaTokenizer`
- ✓ `LLMTokenData` added to `data.py` for storing per-token probability data
- ✓ `ModelInstance.query_log_probs()` implemented using `logprobs=True` chat completion
- ✓ `ModelInstance.get_tokenizer()` added as public accessor for `LlamaTokenizer`
- ✓ Docstring convention changed from NumPy to reStructuredText (pep257 in ruff config)
- ✓ `LLMTokenData.write()` and `read()` implemented; `TokenProbError` added for length mismatch validation
- ✓ `json.loads(reader)` → `json.load(reader)` bug fixed in `LLMTokenData.read()`
- ✓ `cli.py` dataset directories split: `DATA_DIR` replaced by `RESPONSES_DIR` and `PROBS_DIR`
- ✓ `ui_get_probs()` completed; `ui_save_token_data()` added; `ui_select_function()` extracted from `main()`
- ✓ `main()` reordered: function selection now happens before model load so `logits_all` is known at construction time
- ✓ `ModelInstance.__init__` accepts `logits_all: bool = False`; `logits_all=True` set automatically for probs runs
- ✓ `query_log_probs()` fixed: `top_logprobs=1` required alongside `logprobs=True` for llama_cpp chat handler to forward logprobs to the completion layer
- ✓ `match`/`case` syntax bug fixed in `main()` (bare names are capture patterns, not value comparisons)
- ✓ `LLMNextTokenData.write()` and `read()` implemented (JSON format)
- ✓ `LLMNextTokenData.output_vec` type corrected from `TokenSequence` to `list[int]`; `top_m_tokens` tightened to `dict[str, float]`; `self.written` added
- ✓ `data.py` no longer imports `TokenSequence`; `LLMTokenData` uses `list[str]` directly
- ✓ `sample_from_logprobs()` implemented in `analysis/probabilities.py`; exported from `analysis/__init__.py`
- ✓ `AdjustFn` type alias added to `llama_interface.py`
- ✓ `query_log_probs_next_token` signature updated: `context_tokens: list[int]` (was `list[str]`)
- ✓ `ModelInstance._format_chat_prompt()` implemented
- ✓ `ModelInstance.generate_adjusted()` implemented
- ✓ Unit tests added for all new code: `test_probabilities.py` (7 tests), `TestLLMNextTokenData*` in `test_data.py` (20 tests), `TestQueryLogProbsNextToken` / `TestFormatChatPrompt` / `TestGenerateAdjusted` in `test_llama_interface.py` (22 tests)
