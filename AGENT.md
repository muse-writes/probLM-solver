# probLM-solver Codebase Assessment

## Project Overview
**probLM-solver** is a Python library built on top of `llama.cpp` that generates and performs statistical analysis on language model outputs.

- **Repository**: https://github.com/muse-writes/probLM-solver
- **Author**: Clio (d.k.johnson@lancaster.ac.uk)
- **Python Version**: >=3.13, <4.0
- **Build System**: uv_build
- **Status**: Version 1.1.0

---

## Directory Structure
```
probLM-solver/
├── src/problm_solver/          # Main source code (installed as `problm_solver`)
│   ├── __init__.py              # Package initialisation (empty docstring)
│   ├── adjust_probs.py          # GenerationContext, AdjustFn, adjustment functions, BranchSampler hierarchy
│   ├── cli.py                   # Command line interface
│   ├── constants.py             # Memory-size constants (KIB, MIB, GIB)
│   ├── llama_interface.py       # llama.cpp wrapper interface
│   ├── data.py                  # LLM output data containers and serialisation
│   ├── utils.py                 # Internal helpers (_as_rng)
│   └── analysis/                # Statistical analysis subpackage
│       ├── __init__.py          # Re-exports public analysis API
│       ├── tokenizer.py         # Tokenizer ABC and implementations
│       └── probabilities.py     # prob_of_token and sample_from_logprobs utilities
├── tests/
│   ├── __init__.py
│   ├── test_import.py           # Basic import test
│   ├── test_data.py             # Tests for data.py
│   ├── test_cli.py              # Tests for cli.py
│   ├── test_llama_interface.py  # Tests for llama_interface.py
│   ├── test_probabilities.py    # Tests for analysis/probabilities.py
│   └── test_adjust_probs.py    # Tests for adjust_probs.py
├── docs/
│   ├── index.md                 # MkDocs landing page (legacy, kept in place)
│   ├── conf.py                  # Sphinx configuration
│   ├── index.rst                # Sphinx landing page with toctree
│   ├── api.rst                  # Sphinx API reference (one automodule section per module)
│   └── Makefile                 # `make html` builds; `make clean` wipes _build/
├── pyproject.toml               # Project configuration & dependencies
├── Dockerfile                   # Multi-stage Docker build (dev + app targets)
├── docker-compose.yml           # Compose configuration
├── mkdocs.yml                   # MkDocs config (superseded by Sphinx; retained)
├── README.md                    # Project README
└── .gitignore                   # Standard Python gitignore
```

---

## Core Modules

### 1. **adjust_probs.py**
**Purpose**: Defines the `GenerationContext` dataclass, the `AdjustFn` interface, and concrete adjustment function implementations for use with `generate_adjusted()`.

**Type Alias**:
- `AdjustFn = Callable[[GenerationContext], dict[str, float]]`: Receives a `GenerationContext`; returns a modified log-prob dict. Values need not be normalised.

**Dataclass**:
- `GenerationContext`: Bundles everything an adjustment function might need at each generation step.
  - `token_probs: dict[str, float]`: Current top-M candidates and their log-probs
  - `prev_probs: list[float]`: History of normalised probabilities of selected tokens so far
  - `context_tokens: list[int]`: Current token ID sequence
  - `query_next: Callable[[list[int]], dict[str, float] | None]`: Pre-bound query for top-M next-token log-probs; returns `None` on EOS
  - `query_branch: Callable[[list[int], int], float]`: Generates a complete branch of up to `depth` tokens from the given context in a single model call; returns sum of per-token log-probs; returns `0.0` on immediate EOS
  - `tokenize_token: Callable[[str], list[int]]`: Converts a single token string to token ID(s)

**Functions**:
- `adjust_identity(context: GenerationContext) -> dict[str, float]`: No-op; returns `context.token_probs` unchanged.

**Classes**:
- `SampleLowTemp`: Callable class implementing low-temperature power-scaling.
  - `__init__(alpha: float)`: Stores scaling exponent.
  - `__call__(context: GenerationContext) -> dict[str, float]`: Raises current token probabilities to `alpha`, multiplies by the product of all previous token probabilities each also raised to `alpha`, returns as log-probabilities.

- `BranchSampler` (ABC): Abstract base class for branch-level MCMC sampling strategies.
  - `reset() -> None`: Concrete no-op; stateful subclasses override this to clear chain state between candidate-token chains.
  - `step(proposed_log_prob, alpha, forward_log_q, reverse_log_q) -> float`: Abstract; processes one proposed branch and returns the accepted chain state's log-prob.
  - `should_continue(branch_log_probs: npt.NDArray[np.float64]) -> bool`: Abstract; returns `True` if more branches should be sampled.

- `MetropolisSampler(BranchSampler)`: Metropolis-Hastings sampler over complete branch proposals.
  - `__init__(equil_branches=5, max_branches=30, tolerance=1e-1, rng=None)`: `rng` is normalised via `_as_rng()` so an int seed or `None` is accepted. `equil_branches` sets the burn-in period discarded before SEM is computed.
  - `reset()`: Clears `self._current_log_prob` to `None`.
  - `step(...)`: Computes `log_accept_ratio = (alpha-1)*(log p' - log p) + log q(x|x') - log q(x'|x)`; accepts with `self._rng.random()`.
  - `should_continue(branch_log_probs)`: Returns `True` while `n < equil_branches`; `False` when `n >= max_branches`; otherwise uses SEM on `branch_log_probs[equil_branches:]` (equilibration period discarded) vs `tolerance`.

- `SamplePowerDist`: Callable class implementing power-distribution sampling via MCMC lookahead.
  - `__init__(alpha, lookahead_depth, branch_sampler: BranchSampler)`: Injects a `BranchSampler` instance.
  - `__call__(context: GenerationContext) -> dict[str, float]`: For each candidate token, appends its token IDs to the context, then runs `branch_sampler` steps (calling `context.query_branch` once per step) until `should_continue` returns `False`; accumulates branch log-probs into a numpy array and combines with the candidate's own log-prob scaled by `alpha` via log-sum-exp.

**Dependencies**: `abc`, `collections.abc`, `dataclasses`, `numpy`, `problm_solver.utils`, `problm_solver.analysis.probabilities`

---

### 2. **utils.py**
**Purpose**: Internal helpers shared across modules. Not part of the public API.

**Functions**:
- `_as_rng(rng: np.random.Generator | int | None) -> np.random.Generator`: Normalises a seed, `None`, or passes forward an existing generator. Enables injectable, reproducible randomness throughout the package.

**Dependencies**: `numpy`

---

### 3. **constants.py**
**Purpose**: Package-wide numeric constants.

**Constants**:
- `KIB = 1024`, `MIB = KIB * KIB`, `GIB = MIB * KIB`: Memory-size multipliers.

**Dependencies**: none

---

### 4. **data.py**
**Purpose**: Standalone data containers for LLM output; handles serialisation and deserialisation.

**Type Aliases** (used by `LLMOutputDataFull`):
- `Tokens = list[str]`
- `TopKProbs = dict[str, float]`
- `ResponseProbs = tuple[Tokens, list[float]]`: Positionally-aligned pair of chosen token strings and their selection probabilities.
- `TopKResponse = tuple[Tokens, list[TopKProbs]]`: Positionally-aligned pair of chosen token strings and their full adjusted top-k distributions.

**Key Classes**:
- `LLMOutputData`: Stores a prompt and its associated LLM responses.
  - `__init__(prompt: str, data: npt.NDArray[Any])`: Constructed with raw data.
  - `write(fname: str)`: Saves to JSONL; each line has `id`, `prompt`, `response`; sets `self.written = True`. `:param fname:` absolute path.
  - `read(fname: str)`: Loads from JSONL; sets `self.written = True`. `:param fname:` absolute path.
  - `self.written`: Tracks whether in-memory data is in sync with disk.

- `LLMTokenData`: Stores a single tokenized LLM response paired with per-token probabilities.
  - `__init__(prompt: str, tokens: list[str], probs: list[float])`: Validates `len(tokens) == len(probs)`, raising `TokenProbError` otherwise.
  - `write(fname: str)` / `read(fname: str)`: JSON format, single record with `prompt`, `tokens`, `probs`.

- `TokenProbError(ValueError)`: Raised by `LLMTokenData.__init__` when `len(tokens) != len(probs)`.
  - `__init__(data_class: type)`: Formats an error message including the class name.

- `LLMNextTokenData`: Stores the top-M most likely next tokens and their log-probabilities.
  - `__init__(prompt: str, output_vec: list[int], top_m_tokens: dict[str, float])`.
  - `write(fname: str)` / `read(fname: str)`: JSON format with `prompt`, `output_vec`, `top_m_tokens`.
  - `self.written`: Tracks sync with disk.

- `Hyperparams`: Dataclass storing generation hyperparameters alongside an `LLMOutputDataFull`.
  - Fields: `alpha: float`, `top_k: int | None`, `top_p: int | None`, `max_tokens: int`.

- `LLMOutputDataFull`: Richer output dataclass returned by `generate_adjusted()`; records the full adjusted generation process.
  - Fields: `context: Tokens`, `hyperparams: Hyperparams`, `response_probabilities: ResponseProbs`, `response_topk: TopKResponse`, `sampling_method: str`, `branch_sampler: str | None`, `_written: bool` (excluded from serialisation).
  - `write(fname: str)`: Serialises to single-record JSON via `asdict` with `_dict_factory` to exclude `_written`; sets `_written = True`.
  - `read(fname: str)`: Deserialises from JSON; explicitly reconstructs `response_probabilities` and `response_topk` as tuples (JSON returns lists); sets `_written = True`.
  - `_dict_factory(x)`: Static method; excludes `_written` key from `asdict` output.

**Dependencies**: `json`, `dataclasses`, `numpy`

---

### 5. **llama_interface.py**
**Purpose**: Wrapper around `llama_cpp` for local model inference. Owns the responsibility of generating all data container types.

**Key Classes**:
- `ModelInstance`: Keeps a model instance and its context.
  - `__init__(fname: str, context: str, logits_all: bool = False)`: Loads model via `Llama`; `n_ctx=2048` (hard-coded); `logits_all` forwarded to `Llama`. Computes RAM cache capacity from model metadata (`n_ctx × 2 × n_layers × n_kv_heads × head_dim × 2` × 4 states); attaches a `LlamaRAMCache`.
  - `query() -> str`: Single inference call; `max_tokens=512` (hard-coded); chat completion with `role='user'`.
  - `query_n_times(n: int) -> npt.NDArray[Any]`: Calls `query()` N times.
  - `generate_data(n_samples: int) -> LLMOutputData`.
  - `query_log_probs() -> LLMTokenData`: Chat completion with `logprobs=True, top_logprobs=1`; extracts tokens and `exp(logprob)` probabilities.
  - `query_log_probs_next_token(context_tokens: list[int], n_tokens: int) -> LLMNextTokenData | None`: `create_completion` with `max_tokens=1, logprobs=n_tokens`; returns `None` on EOS (empty `top_logprobs`).
  - `query_branch(context_tokens: list[int], max_tokens: int) -> float`: Single `create_completion` call with `logprobs=1`; sums `token_logprobs`; returns `0.0` on immediate EOS.
  - `reset_state() -> None`: Calls `self._llm.reset()`. Low-level API stub.
  - `eval_tokens(tokens: list[int]) -> None`: Calls `self._llm.eval(tokens)`. Low-level API stub.
  - `save_live_state() -> LlamaState`: Calls `self._llm.save_state()`. Low-level API stub (see Known Issues).
  - `load_live_state(state: LlamaState) -> None`: Calls `self._llm.load_state(state)`. Low-level API stub.
  - `get_tokenizer() -> LlamaTokenizer`: Returns a `LlamaTokenizer` backed by `self._llm`.
  - `_format_chat_prompt() -> list[int]`: Builds prompt via `Jinja2ChatFormatter` from `metadata['tokenizer.chat_template']`; tokenises to `list[int]`.
  - `_tokens_as_strings(token_ids: list[int]) -> list[str]`: Converts each token ID to its string representation via `detokenize([tid], special=True)`.
  - `generate_adjusted(n_tokens: int, adjust_fn: AdjustFn, max_tokens: int, *, alpha: float = 1.0, top_p: int | None = None, sampling_method: str | None = None, branch_sampler: str | None = None) -> LLMOutputDataFull`: Token-by-token loop; builds a `GenerationContext` at each step (with pre-bound `query_next`, `query_branch`, and `tokenize_token` callables); passes it to `adjust_fn`; accumulates chosen tokens, their selection probabilities, and adjusted top-k distributions; casts all floats to Python-native; breaks on `None`, EOS token ID, or empty token ID list; constructs and returns `LLMOutputDataFull`. `sampling_method` is auto-derived from `adjust_fn`'s class name if not supplied.

**Dependencies**: `copy`, `math`, `inspect`, `llama_cpp`, `llama_cpp.llama_chat_format.Jinja2ChatFormatter`, `numpy`, `problm_solver.adjust_probs`, `problm_solver.data`, `problm_solver.analysis.tokenizer`, `problm_solver.analysis.probabilities`

---

### 6. **cli.py**
**Purpose**: Interactive command-line interface for the full generate-and-save workflow.

**Constants**:
- `PROBLM_DIR`: `~/.problm-solver/`
- `MODELS_DIR`: `~/.problm-solver/models/`
- `RESPONSES_DIR`: `~/.problm-solver/datasets/responses/`
- `PROBS_DIR`: `~/.problm-solver/datasets/probabilities/`
- `NUMBER_OF_FUNCTIONS = 4`, `GEN_DATA = 1`, `PROBS = 2`, `LOW_TEMP = 3`, `POWER_SAMPLING = 4`

**Functions**:
- `ensure_models_dir() / ensure_responses_dir() / ensure_probs_dir()`: Create directories if absent; return their path.
- `list_models() -> list[Path]`: Sorted list of `.gguf` files in `MODELS_DIR`.
- `get_responses_path / get_adjusted_path / get_probs_path`: Timestamped output paths.
- `ui_select_model() -> Path`: Interactive model picker.
- `ui_select_function() -> int`: Returns 1–4.
- `ui_gen_data(model, model_path)`: Prompts for sample count; delegates to `ui_save_data`.
- `ui_save_data(fname: str, data: LLMOutputData)`: Prompts user to confirm; on `'y'` calls `data.write(fname)`; on `'n'` prompts for an alternative filename (empty string intended to discard — see Known Issues).
- `ui_save_token_data(fname: str, data: LLMTokenData)`: Prompts user to confirm; calls `data.write()`.
- `ui_get_probs(model, model_path)`: Calls `model.query_log_probs()`; delegates to `ui_save_token_data`.
- `ui_generate_low_temp(model, model_path)`: Prompts for `alpha`, `top_k`, `max_tokens`; constructs `SampleLowTemp(alpha)`; calls `model.generate_adjusted()` passing `alpha`; delegates save to `ui_save_data`.
- `ui_generate_power_mcmc(model, model_path)`: Prompts for `alpha`, lookahead depth, `top_k`, `max_tokens`; constructs `SamplePowerDist(alpha, lookahead_depth, MetropolisSampler(max_branches=10))`; calls `model.generate_adjusted()` passing `alpha`, `sampling_method='Power Distribution'`, `branch_sampler='Metropolis Sampler'`; delegates save to `ui_save_data`.
- `main()`: Select model → enter prompt → select function → load model (`logits_all=True` for `PROBS`, `LOW_TEMP`, `POWER_SAMPLING`) → dispatch via `match`.

**Dependencies**: `problm_solver.adjust_probs`, `problm_solver.llama_interface`, `problm_solver.data`

---

### 7. **analysis/\_\_init\_\_.py**
Re-exports the public API of the `analysis` subpackage:
`Tokenizer`, `WordTokenizer`, `LlamaTokenizer`, `Token`, `TokenSequence`, `prob_of_token`, `sample_from_logprobs`

---

### 8. **analysis/tokenizer.py**
**Purpose**: Tokenizer abstraction and implementations for splitting LLM response strings.

**Type Aliases**: `Token = str`, `TokenSequence = list[Token]`

**Classes**:
- `Tokenizer` (ABC): `tokenize(text: str) -> TokenSequence`
- `WordTokenizer(Tokenizer)`: Regex-based; `__init__(*, lowercase: bool = False)`
- `LlamaTokenizer(Tokenizer)`: BPE tokenizer backed by a `llama_cpp.Llama` instance; `__init__(llama: _LlamaInterface)`
- `_LlamaInterface` (Protocol, private): Structural protocol for `tokenize` / `detokenize`

**Dependencies**: `re`, `abc`, `typing`

---

### 9. **analysis/probabilities.py**
**Purpose**: Token probability utility functions.

**Functions**:
- `prob_of_token(token: str, log_probs: dict[str, float]) -> float`: Normalised probability of a specific token from a log-prob dict.
- `sample_from_logprobs(log_probs: dict[str, float], rng: np.random.Generator | int | None = None) -> str`: Samples a token string from a log-probability distribution via shift-exp-normalise then `rng.choice()`.

**Dependencies**: `numpy`, `problm_solver.utils`

---

## Architecture

### Dependency Direction
```
cli.py → ModelInstance (llama_interface.py) → LLMOutputData / LLMTokenData / LLMNextTokenData / LLMOutputDataFull / Hyperparams (data.py)
                                             → LlamaTokenizer (analysis/tokenizer.py)
                                             → prob_of_token, sample_from_logprobs (analysis/probabilities.py)
                                             → AdjustFn, GenerationContext (adjust_probs.py)
cli.py → SampleLowTemp, SamplePowerDist, MetropolisSampler (adjust_probs.py)
cli.py → LLMOutputData (data.py)
adjust_probs.py → _as_rng (utils.py)
adjust_probs.py → prob_of_token (analysis/probabilities.py)
analysis/probabilities.py → _as_rng (utils.py)
analysis/tokenizer.py → (no intra-package deps)
data.py → (no intra-package deps)
utils.py → (no intra-package deps)
constants.py → (no intra-package deps)
```

### Data Flow
1. User selects a GGUF model file from `~/.problm-solver/models/`
2. User enters a prompt, then selects a function (1–4)
3. `ModelInstance` is created; `logits_all=True` for functions 2, 3, and 4
4. **Text responses** (1): `model.generate_data(n)` → `LLMOutputData`; saved to `responses/`
5. **Token + probability responses** (2): `model.query_log_probs()` → `LLMTokenData`; saved to `probabilities/`
6. **Low-temperature generation** (3): `model.generate_adjusted(n_tokens, SampleLowTemp(alpha), max_tokens, alpha=alpha)` → `LLMOutputDataFull`; saved to `responses/`
7. **Power MCMC generation** (4): `model.generate_adjusted(n_tokens, SamplePowerDist(alpha, depth, MetropolisSampler()), max_tokens, alpha=alpha, sampling_method=..., branch_sampler=...)` → `LLMOutputDataFull`; saved to `responses/`

---

## Dependencies

### Runtime (`dependencies` in pyproject.toml)
- `llama-cpp-python`: LLM inference via llama.cpp
- `numpy`: Array operations
- `poethepoet`: Task runner (kept runtime so it is available inside the Docker app container)

### Development (`dependency-groups.dev`)
- `pytest >= 8.3.4` + `pytest-mock`, `pytest-xdist`: Testing
- `coverage[toml] >= 7.6.10`: Code coverage (minimum 50%)
- `ruff >= 0.9.2`: Linting & formatting
- `pre-commit >= 4.0.1`: Git hooks
- `commitizen >= 4.3.0`: Semantic versioning
- `sphinx >= 9.1.0` + `sphinx-autodoc-typehints`, `shibuya`: Reference documentation
- `mkdocs-material >= 9.5.21`: MkDocs (superseded by Sphinx; retained)
- `typeguard >= 4.4.1`, `ty >= 0.0.6`: Type checking
- `ipython`, `ipykernel`, `ipywidgets`: Jupyter support
- `codespell >= 2.4.1`: Spell checking

---

## Docker Setup

### Dockerfile (multi-stage)
- **`dev` stage**: Based on `ghcr.io/astral-sh/uv:python3.13-trixie`; sets up venv at `/opt/venv`; intended for VS Code Dev Containers
- **`app` stage**: Based on `python:3.13-slim`; runs `uv sync --no-dev --no-editable --frozen`; venv at `/workspaces/problm-solver/.venv`; ENTRYPOINT is `.venv/bin/poe`, CMD is `serve`

### docker-compose.yml
- **`devcontainer`**: Runs by default; mounts `..:/workspaces`
- **`app`**: Requires explicit profile; `stdin_open: true` and `tty: true`; mounts `~/.problm-solver:/home/user/.problm-solver`

---

## Poe Tasks (`[tool.poe.tasks]`)

```bash
poe serve           # Run the CLI (calls the `problm-solver` entry point)
poe lint            # Run pre-commit hooks on all files
poe test            # Run tests + coverage report + coverage XML
poe docs            # Build Sphinx docs (cd docs && make html)
poe docs --serve    # Open built docs in browser (xdg-open docs/_build/html/index.html)
```

---

## Code Quality & Configuration

### Testing
- **Framework**: pytest with doctest modules
- **Coverage**: Minimum 50% (`fail_under = 50`)
- **Options**: Exit on first failure, verbose (v=2), JUnit XML reports
- **Paths**: `src/`, `tests/`
- **Test files**: `test_import.py`, `test_data.py`, `test_cli.py`, `test_llama_interface.py`, `test_probabilities.py`, `test_adjust_probs.py`
- **Total tests**: 210
- **Doctests**: `WordTokenizer` and `LlamaTokenizer` have inline doctests collected by `--doctest-modules`
- **Mocking**:
  - `llama_cpp.Llama` is patched at construction time in all `ModelInstance` tests; `_make_llama_mock` provides metadata and `n_ctx` so `__init__`'s cache-sizing logic runs correctly
  - `llama_cpp.LlamaRAMCache` is patched in `test_cache_sized_from_model_metadata`
  - `Jinja2ChatFormatter` is patched at `problm_solver.llama_interface.Jinja2ChatFormatter` in `TestFormatChatPrompt`
  - `_format_chat_prompt`, `query_log_probs_next_token`, `sample_from_logprobs`, and `prob_of_token` are all patched in the `gen_adj_model` fixture via `contextlib.ExitStack`
  - `adjust_fn` receives a `GenerationContext`; `MagicMock.call_args_list` inspection reads `.prev_probs` from the context object
  - `GenerationContext` fixtures in `test_adjust_probs.py` supply a `query_branch=MagicMock(return_value=-1.5)` field
  - CLI filesystem interactions use `monkeypatch` against `tmp_path`

### Linting (Ruff)
- **Line Length**: 100 characters
- **Target Python**: 3.13
- **Docstring Convention**: reStructuredText / PEP 257 (`:param:`, `:returns:` directives)
- **Quotes**: Single (inline and multiline)
- **Imports**: Absolute only (`ban-relative-imports = "all"`)

### Versioning
- Conventional Commits enforced via Commitizen
- Current version: **1.1.0**

---

## Known Issues — Active

1. **`ui_save_data()` discard logic unreachable** (`cli.py`): `input()` always returns a `str`; `if fname is not None` is always `True`, so the `print('Aborted saving.')` branch can never execute. Should be `if fname` (truthy check for empty string).

2. **Unreachable guard in `ui_gen_data()`** (`cli.py`): `data_size = int(input(...))` either succeeds (always an `int`) or raises; the `while not isinstance(data_size, int)` loop below it can never be reached.

3. **Hard-coded model parameters** (`llama_interface.py`): `n_ctx=2048` and `max_tokens=512` are not user-configurable.

4. **`FBT001` on `_LlamaInterface` protocol** (`analysis/tokenizer.py`): `add_bos: bool` and `special: bool` are flagged as boolean positional arguments. These mirror the external `llama_cpp.Llama` API and cannot be made keyword-only.

5. **Equilibration period not user-configurable** (`adjust_probs.py`): `should_continue()` discards `branch_log_probs[:equil_branches]` as a fixed equilibration period. Whether this is the right threshold, and whether users should be able to control it separately from `equil_branches`, is an open design question (TODO in source).

6. **`save_live_state()` missing return** (`llama_interface.py`): calls `self._llm.save_state()` but does not return its result; annotated `-> LlamaState` but always returns `None`.

7. **`top_p` stubbed** (`llama_interface.py`): `top_p` keyword argument accepted by `generate_adjusted()` but raises `NotImplementedError` if passed any non-`None` value.

8. **`ui_generate_low_temp` missing `sampling_method`** (`cli.py`): does not pass `sampling_method` to `generate_adjusted()`; the label stored in `LLMOutputDataFull` is auto-derived as `'SampleLowTemp'` rather than a clean human-readable string.

9. **`get_adjusted_path()` extension** (`cli.py`): returns a `.jsonl` path, but `LLMOutputDataFull.write()` produces single-record JSON; should be `.json`.

10. **`_written` comment indentation** (`data.py`): the inline comment `# Unsaved data state tracking variable.` before `_written` is at column 0 inside the `LLMOutputDataFull` class body; should be indented 4 spaces.

---

## Known Issues — Resolved

- ✓ `LLMOutputData.read()` implemented
- ✓ `LLMOutputData` dependency on `ModelInstance` removed; `generate_data()` added to `ModelInstance`
- ✓ `llama-cpp-python` and `numpy` added to runtime dependencies
- ✓ `poethepoet` moved to runtime dependencies for Docker app container compatibility
- ✓ Poe executor changed from `simple` to `virtualenv`
- ✓ `poe serve` wired up to `problm-solver` entry point (was a placeholder echo)
- ✓ Intra-package imports converted from bare names to absolute (`problm_solver.*`)
- ✓ Docker `app` service configured with `stdin_open`, `tty`, and volume mount
- ✓ Test file bug fixed (`import problm-solver` → `import problm_solver`)
- ✓ Test suite created: `test_data.py`, `test_cli.py`, `test_llama_interface.py`
- ✓ `analysis/` subpackage created with `Tokenizer` ABC, `WordTokenizer`, `LlamaTokenizer`
- ✓ `LLMTokenData` added to `data.py`
- ✓ `ModelInstance.query_log_probs()` implemented
- ✓ `ModelInstance.get_tokenizer()` added
- ✓ Docstring convention changed from NumPy to reStructuredText
- ✓ `LLMTokenData.write()` / `read()` implemented; `TokenProbError` added
- ✓ `json.loads(reader)` → `json.load(reader)` bug fixed in `LLMTokenData.read()`
- ✓ `cli.py` dataset directories split: `DATA_DIR` → `RESPONSES_DIR` + `PROBS_DIR`
- ✓ `ui_get_probs()` completed; `ui_save_token_data()` added; `ui_select_function()` extracted
- ✓ `main()` reordered: function selection before model load so `logits_all` is known at construction
- ✓ `query_log_probs()` fixed: `top_logprobs=1` required alongside `logprobs=True`
- ✓ `match`/`case` syntax bug fixed in `main()` (bare names are capture patterns)
- ✓ `LLMNextTokenData.write()` / `read()` implemented
- ✓ `LLMNextTokenData.output_vec` type corrected to `list[int]`; `self.written` added
- ✓ `sample_from_logprobs()` implemented; exported from `analysis/__init__.py`
- ✓ `AdjustFn` type alias added and moved to `adjust_probs.py`
- ✓ `query_log_probs_next_token` signature corrected to `context_tokens: list[int]`
- ✓ `ModelInstance._format_chat_prompt()` implemented via `Jinja2ChatFormatter`
- ✓ `ModelInstance.generate_adjusted()` implemented
- ✓ `query_log_probs_next_token` EOS crash fixed (empty `top_logprobs` → `None`)
- ✓ `logprobs` parameter corrected from `True` to `n_tokens` (int) in `create_completion`
- ✓ Mutable `prev_probs` argument bug fixed: defensive copy passed to `adjust_fn`
- ✓ `Probabilities` class stub deleted
- ✓ `prob_of_token()` added; exported from `analysis/__init__.py`
- ✓ `adjust_probs.py` created: `AdjustFn`, `adjust_identity`, `SampleLowTemp`
- ✓ `ui_generate_adjusted()` / `get_adjusted_path()` added to `cli.py`
- ✓ `raise UnexpectedFunctionError` → `raise UnexpectedFunctionError()` fixed
- ✓ Unit tests added: `test_adjust_probs.py`; `TestProbOfToken`; new `TestQueryLogProbsNextToken`, `TestFormatChatPrompt`, `TestGenerateAdjusted`
- ✓ Version bumped to 1.0.0
- ✓ `GenerationContext` dataclass added to `adjust_probs.py`; `AdjustFn` updated to `Callable[[GenerationContext], dict[str, float]]`
- ✓ `BranchSampler` ABC and `MetropolisSampler` implemented in `adjust_probs.py`
- ✓ `SamplePowerDist` implemented in `adjust_probs.py`
- ✓ `generate_adjusted()` updated to build and inject `GenerationContext` at each step
- ✓ `_as_rng()` helper added in new `utils.py`; legacy global RNG eliminated from `MetropolisSampler` and `sample_from_logprobs`
- ✓ SEM equilibration bug fixed in `MetropolisSampler.should_continue`: warmup samples now discarded via `branch_log_probs[equil_branches:]`
- ✓ `ui_generate_power_mcmc()` added to `cli.py`; `POWER_SAMPLING = 4` constant added; `main()` dispatch updated
- ✓ `ui_generate_adjusted()` renamed to `ui_generate_low_temp()`; `GENERATE_ADJUSTED` renamed to `LOW_TEMP`
- ✓ Sphinx reference docs added: `docs/Makefile`, `docs/conf.py`, `docs/index.rst`, `docs/api.rst`; `sphinx`, `sphinx-autodoc-typehints`, `shibuya` added to dev dependencies; `poe docs` task migrated from MkDocs to Sphinx
- ✓ `ModelInstance.__init__` now computes RAM cache capacity from model metadata; `LlamaRAMCache` created and attached
- ✓ `ModelInstance.query_branch()` added: single-call branch log-prob evaluation replacing token-by-token `query_next` loop in `SamplePowerDist`
- ✓ Low-level API stubs added to `ModelInstance`: `reset_state()`, `eval_tokens()`, `save_live_state()`, `load_live_state()`
- ✓ `GenerationContext.query_branch` field added; `SamplePowerDist.__call__` updated to call it instead of looping over `query_next`
- ✓ `MetropolisSampler` parameter renamed `min_branches` → `equil_branches`; `max_branches` default reduced 50 → 30
- ✓ `Hyperparams` and `LLMOutputDataFull` dataclasses added to `data.py`; `ResponseProbs` type changed from `dict[str, float]` to `tuple[Tokens, list[float]]` to preserve duplicate token positions
- ✓ `LLMOutputDataFull.read()` tuple reconstruction fixed: `response_probabilities` and `response_topk` now explicitly rebuilt as tuples after JSON deserialisation (was left as lists)
- ✓ `generate_adjusted()` return type changed from `LLMOutputData` to `LLMOutputDataFull`; new keyword-only params `alpha`, `top_p`, `sampling_method`, `branch_sampler` added
- ✓ `numpy` float serialisation bug fixed: `adjusted` dict values and `prob_of_token` results cast to `float()` at accumulation time in `generate_adjusted()`
- ✓ `AttributeError` on `adjust_fn.__name__` fixed: callable class instances handled via `isclass`/`getattr`/`type()` fallback
- ✓ `constants.py` added with `KIB`, `MIB`, `GIB`
- ✓ `_make_llama_mock` updated with model metadata and `n_ctx`; `test_cache_sized_from_model_metadata` added; `TestModelInstanceQueryBranch` test class added
- ✓ `test_adjust_probs.py` updated: `query_branch` added to all `GenerationContext` fixtures; `SamplePowerDist` tests rewritten around `query_branch`; `MetropolisSampler` tests updated to `equil_branches`
