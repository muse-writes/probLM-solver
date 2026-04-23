# probLM-solver Codebase Assessment

## Project Overview
**probLM-solver** is a Python library built on top of `llama.cpp` that generates and performs statistical analysis on language model outputs.

- **Repository**: https://github.com/muse-writes/probLM-solver
- **Author**: Clio (d.k.johnson@lancaster.ac.uk)
- **Python Version**: >=3.13, <4.0
- **Build System**: uv_build
- **Status**: Version 1.0.0

---

## Directory Structure
```
probLM-solver/
├── src/problm_solver/          # Main source code (installed as `problm_solver`)
│   ├── __init__.py              # Package initialisation (empty docstring)
│   ├── adjust_probs.py          # AdjustFn type alias and adjustment function implementations
│   ├── cli.py                   # Command line interface
│   ├── llama_interface.py       # llama.cpp wrapper interface
│   ├── data.py                  # LLM output data containers and serialisation
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
│   ├── test_probabilities.py   # Tests for analysis/probabilities.py
│   └── test_adjust_probs.py    # Tests for adjust_probs.py
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

### 1. **adjust_probs.py**
**Purpose**: Defines the `AdjustFn` interface and provides concrete adjustment function implementations for use with `generate_adjusted()`.

**Module-level**:
- `AdjustFn = Callable[[dict[str, float], list[float]], dict[str, float]]`: Type alias for a probability adjustment callable. Receives the top-M `{token: log_prob}` dict and a copy of `prev_probs` (normalised probabilities of all previously selected tokens in this generation); returns a modified log-prob dict. Values need not be normalised.

**Functions**:
- `adjust_identity(token_probs, prev_probs) -> dict[str, float]`: No-op; returns `token_probs` unchanged. Satisfies the `AdjustFn` interface without modifying the distribution. Useful as a baseline and for testing.

**Classes**:
- `SampleLowTemp`: Callable class implementing low-temperature power-scaling
  - `__init__(alpha: float)`: Stores scaling exponent. Values > 1 sharpen the distribution; values between 0 and 1 flatten it.
  - `__call__(token_probs, prev_probs) -> dict[str, float]`: Raises current token probabilities to `alpha`, multiplies by the product of all previous token probabilities each also raised to `alpha`, and returns as log-probabilities.

**Dependencies**: `collections.abc.Callable`, `numpy`

---

### 2. **data.py**
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

- `TokenProbError(ValueError)`: Raised by `LLMTokenData.__init__` when `len(tokens) != len(probs)`

- `LLMNextTokenData`: Stores the top-M most likely next tokens and their log-probabilities given a prompt and current output vector
  - `__init__(prompt: str, output_vec: list[int], top_m_tokens: dict[str, float])`: `output_vec` is the current token ID sequence; `top_m_tokens` maps token strings to log-probs
  - `write(fname: str)`: Saves to JSON format with `prompt`, `output_vec`, `top_m_tokens` fields; sets `self.written = True`
  - `read(fname: str)`: Loads from JSON; populates all three fields; sets `self.written = True`
  - `self.written`: Tracks whether in-memory data is in sync with disk

**Dependencies**: `json`, `numpy`

---

### 3. **llama_interface.py**
**Purpose**: Wrapper around `llama_cpp` for local model inference. Owns the responsibility of generating all data container types.

**Key Classes**:
- `ModelInstance`: Represents a loaded GGUF model
  - `__init__(fname: str, context: str, logits_all: bool = False)`: Loads model via `Llama`; hard-coded `n_ctx=2048`; `logits_all` forwarded to `Llama`
  - `query() -> str`: Single inference call; hard-coded `max_tokens=512`; uses chat completion with `role='user'`
  - `query_n_times(n: int) -> npt.NDArray[Any]`: Calls `query()` N times, returns numpy array
  - `generate_data(n_samples: int) -> LLMOutputData`: Calls `query_n_times()` and wraps result in `LLMOutputData`
  - `query_log_probs() -> LLMTokenData`: Single inference call with `logprobs=True, top_logprobs=1`; extracts token strings and probabilities (`exp(logprob)`); returns `LLMTokenData`
  - `query_log_probs_next_token(context_tokens: list[int], n_tokens: int) -> LLMNextTokenData | None`: Calls `create_completion()` with `max_tokens=1` and `logprobs=n_tokens` (integer — `create_completion` takes `logprobs: Optional[int]`, not a boolean); returns `None` when `top_logprobs` is empty, which indicates EOS was generated (llama_cpp omits EOS from logprob output)
  - `get_tokenizer() -> LlamaTokenizer`: Returns a `LlamaTokenizer` backed by this model's `Llama` instance
  - `_format_chat_prompt() -> list[int]`: Constructs a `Jinja2ChatFormatter` from `self._llm.metadata['tokenizer.chat_template']` with EOS/BOS token strings from the model, applies it to `self.context` as a user-role message, and tokenises the result to a `list[int]`
  - `generate_adjusted(n_tokens: int, adjust_fn: AdjustFn, max_tokens: int) -> LLMOutputData`: Token-by-token generation loop; maintains `prev_probs: list[float]` (reset each call); passes a copy of `prev_probs` to `adjust_fn` at each step; records chosen token's normalised probability via `prob_of_token`; breaks on `None` from `query_log_probs_next_token`, EOS token ID, or empty token ID list

**Dependencies**: `math`, `llama_cpp`, `llama_cpp.llama_chat_format.Jinja2ChatFormatter`, `numpy`, `problm_solver.adjust_probs`, `problm_solver.data`, `problm_solver.analysis.tokenizer`, `problm_solver.analysis.probabilities`

---

### 4. **cli.py**
**Purpose**: Interactive command-line interface for the full generate-and-save workflow.

**Constants**:
- `PROBLM_DIR`: `~/.problm-solver/`
- `MODELS_DIR`: `~/.problm-solver/models/`
- `RESPONSES_DIR`: `~/.problm-solver/datasets/responses/`
- `PROBS_DIR`: `~/.problm-solver/datasets/probabilities/`
- `NUMBER_OF_FUNCTIONS = 3`, `GEN_DATA = 1`, `PROBS = 2`, `GENERATE_ADJUSTED = 3`

**Functions**:
- `ensure_models_dir() / ensure_responses_dir() / ensure_probs_dir()`: Create directories if absent; each returns its path
- `list_models() -> list[Path]`: Sorted list of `.gguf` files in `MODELS_DIR`
- `get_responses_path(model_path: Path) -> Path`: Timestamped `.jsonl` path inside `RESPONSES_DIR`
- `get_adjusted_path(model_path: Path) -> Path`: Timestamped `adjusted_*.jsonl` path inside `RESPONSES_DIR`
- `get_probs_path(model_path: Path) -> Path`: Timestamped `prob_*.json` path inside `PROBS_DIR`
- `ui_select_model() -> Path`: Interactive model picker
- `ui_select_function() -> int`: Interactive function picker; returns 1, 2, or 3
- `ui_gen_data(model, model_path)`: Prompts for sample count, calls `model.generate_data()`, delegates to `ui_save_data`
- `ui_save_data(fname: str, data: LLMOutputData)`: Prompts user to confirm saving; calls `data.write()`
- `ui_save_token_data(fname: str, data: LLMTokenData)`: Prompts user to confirm saving; calls `data.write()`
- `ui_get_probs(model, model_path)`: Calls `model.query_log_probs()`, delegates to `ui_save_token_data`
- `ui_generate_adjusted(model, model_path)`: Prompts for `alpha` (float), `top_m` (int), `max_tokens` (int); constructs `SampleLowTemp(alpha)`; calls `model.generate_adjusted()`; delegates to `ui_save_data`
- `main()`: Full workflow — select model → enter prompt → select function → load model (`logits_all=True` for `PROBS` or `GENERATE_ADJUSTED`) → dispatch

**Dependencies**: `problm_solver.adjust_probs`, `problm_solver.llama_interface`, `problm_solver.data`

---

### 5. **analysis/\_\_init\_\_.py**
Re-exports the public API of the `analysis` subpackage:
`Tokenizer`, `WordTokenizer`, `LlamaTokenizer`, `Token`, `TokenSequence`, `prob_of_token`, `sample_from_logprobs`

---

### 6. **analysis/tokenizer.py**
**Purpose**: Tokenizer abstraction and implementations for splitting LLM response strings into token sequences for statistical analysis.

**Type Aliases**:
- `Token = str`: A single decoded token string
- `TokenSequence = list[Token]`: An ordered list of tokens from one response

**Key Classes**:
- `Tokenizer` (ABC): Abstract base class; defines `tokenize(text: str) -> TokenSequence`
- `WordTokenizer(Tokenizer)`: Regex-based tokenizer; no external dependencies
  - Splits on word boundaries; keeps contractions; punctuation becomes individual tokens; whitespace discarded
  - `__init__(*, lowercase: bool = False)`
- `LlamaTokenizer(Tokenizer)`: Tokenizer backed by a loaded `llama_cpp.Llama` instance
  - Uses the model's own BPE vocabulary; each token ID decoded individually via `detokenize([id])`
  - `__init__(llama: _LlamaInterface)`
- `_LlamaInterface` (Protocol, private): Structural protocol for `tokenize` and `detokenize` methods

**Dependencies**: `re`, `abc`, `typing`

---

### 7. **analysis/probabilities.py**
**Purpose**: Token probability utility functions.

**Functions**:
- `prob_of_token(token: str, log_probs: dict[str, float]) -> float`: Returns the normalised probability of a specific token from a log-prob dict. Same shift-exp-normalise procedure as `sample_from_logprobs`; used by `generate_adjusted()` to record `prev_probs` after each step.
- `sample_from_logprobs(log_probs: dict[str, float]) -> str`: Samples a token string from a log-probability distribution. Shifts by max before `exp()` for numerical stability, renormalises, samples via `np.random.choice`.

**Dependencies**: `numpy`

---

## Architecture

### Dependency Direction
```
cli.py → ModelInstance (llama_interface.py) → LLMOutputData (data.py)
                                             → LLMTokenData (data.py)
                                             → LLMNextTokenData (data.py)
                                             → LlamaTokenizer (analysis/tokenizer.py)
                                             → prob_of_token (analysis/probabilities.py)
                                             → sample_from_logprobs (analysis/probabilities.py)
                                             → AdjustFn (adjust_probs.py)
cli.py → SampleLowTemp, adjust_identity (adjust_probs.py)
cli.py → LLMOutputData (data.py)
adjust_probs.py → (no intra-package deps)
analysis/probabilities.py → (no intra-package deps)
analysis/tokenizer.py → (no intra-package deps)
data.py → (no intra-package deps)
```

### Data Flow
1. User selects a GGUF model file from `~/.problm-solver/models/`
2. User enters a prompt, then selects a function (1/2/3)
3. `ModelInstance` is created; `logits_all=True` for functions 2 (`PROBS`) and 3 (`GENERATE_ADJUSTED`)
4. **Text responses**: `model.generate_data(n)` → `LLMOutputData`; saved to `~/.problm-solver/datasets/responses/`
5. **Token + probability responses**: `model.query_log_probs()` → `LLMTokenData`; saved to `~/.problm-solver/datasets/probabilities/`
6. **Adjusted generation**: `model.generate_adjusted(n_tokens, adjust_fn, max_tokens)` drives a token-by-token loop; `adjust_fn` receives the top-M log-prob dict and a copy of `prev_probs`; the next token is sampled and its normalised probability recorded; stops on EOS or `None` from `query_log_probs_next_token` → `LLMOutputData`; saved to `~/.problm-solver/datasets/responses/`

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

---

## Code Quality & Configuration

### Testing
- **Framework**: pytest with doctest modules
- **Coverage**: Minimum 50% (`fail_under = 50`)
- **Options**: Exit on first failure, verbose (v=2), JUnit XML reports
- **Paths**: `src/`, `tests/`
- **Test files**: `test_import.py`, `test_data.py`, `test_cli.py`, `test_llama_interface.py`, `test_probabilities.py`, `test_adjust_probs.py`
- **Doctests**: `WordTokenizer` and `LlamaTokenizer` have inline doctests collected by `--doctest-modules`
- **Mocking**:
  - `llama_cpp.Llama` is patched at construction time in all `ModelInstance` tests
  - `Jinja2ChatFormatter` is patched at `problm_solver.llama_interface.Jinja2ChatFormatter` in `TestFormatChatPrompt`; fixture sets up `metadata`, `token_eos`, `token_bos`, `detokenize` on the mock LLM
  - `_format_chat_prompt`, `query_log_probs_next_token`, `sample_from_logprobs`, and `prob_of_token` are all patched in the `gen_adj_model` fixture via `contextlib.ExitStack`
  - `adjust_fn` receives a copy of `prev_probs` (not the live list), making `MagicMock.call_args_list` inspection reliable
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
- Current version: **1.0.0**

---

## Planned: `GenerationContext` and `SamplePowerDist` — Generalised Lookahead Adjustment

This feature generalises the `AdjustFn` interface so that adjustment functions can optionally sample future token sequences ("branches") when computing their adjustment, enabling techniques such as the power distribution sampling from [arxiv:2510.14901](https://arxiv.org/pdf/2510.14901).

### The limitation

The current `AdjustFn = Callable[[dict[str, float], list[float]], dict[str, float]]` can only express functions of the top-M candidates and past history. It has no access to the model or the current token ID sequence, so lookahead-based adjustment is impossible.

### Step 1 — Add `GenerationContext` dataclass to `adjust_probs.py`

A dataclass bundling everything an adjustment function might need, injected from `generate_adjusted()`. A single context object keeps `AdjustFn` stable as requirements grow.

```python
@dataclass
class GenerationContext:
    token_probs: dict[str, float]                        # current top-M candidates
    prev_probs: list[float]                              # history of selected token probs
    context_tokens: list[int]                            # current token ID sequence
    query_next: Callable[[list[int]], dict[str, float] | None]
    # Queries top-M next-token log-probs for a given context. Pre-bound to
    # current n_tokens. Returns None on EOS.
    tokenize_token: Callable[[str], list[int]]
    # Converts a single token string to its token ID(s).
```

`query_next` and `tokenize_token` are plain callables — no import of `ModelInstance` in `adjust_probs.py`, so no circular dependency.

### Step 2 — Update `AdjustFn`

```python
AdjustFn = Callable[[GenerationContext], dict[str, float]]
```

### Step 3 — Update `adjust_identity` and `SampleLowTemp`

Both signatures change to accept a `GenerationContext`. Logic is unchanged — they read `context.token_probs` and `context.prev_probs` instead of separate positional arguments.

### Step 4 — Update `generate_adjusted()` in `llama_interface.py`

Build a `GenerationContext` at each step and pass it to `adjust_fn` instead of the two-argument call:

```python
ctx = GenerationContext(
    token_probs=next_token_data.top_m_tokens,
    prev_probs=list(prev_probs),
    context_tokens=list(context),  # defensive copy
    query_next=lambda ctx_ids: (
        nd.top_m_tokens
        if (nd := self.query_log_probs_next_token(ctx_ids, n_tokens))
        else None
    ),
    tokenize_token=lambda s: self._llm.tokenize(
        s.encode('utf-8'), add_bos=False, special=False
    ),
)
adjusted = adjust_fn(ctx)
```

### Step 5 — Add `BranchSampler` ABC and `MetropolisSampler` to `adjust_probs.py`

To keep the branch-sampling strategy extensible, define an ABC that `SamplePowerDist` accepts as a constructor argument:

```python
class BranchSampler(ABC):
    @abstractmethod
    def sample(self, log_probs: dict[str, float]) -> str:
        """Sample a single token from a log-prob distribution."""

    def reset(self) -> None:
        """Reset any internal state between branches. No-op for stateless samplers."""
```

`MetropolisSampler(BranchSampler)` implements Metropolis-Hastings: at each step it proposes a token from the model distribution and accepts it with probability `min(1, p(proposed) / p(current))`. It is stateful (tracks the current accepted token), so `reset()` clears this state between branches.

Future sampling strategies (e.g. greedy, temperature-scaled) can be added as further `BranchSampler` subclasses without touching `SamplePowerDist`.

### Step 6 — Implement `SamplePowerDist` in `adjust_probs.py`

New callable class implementing the power distribution. For each candidate token, samples `n_branches` future paths of length `lookahead_depth` using the injected `BranchSampler`, accumulates branch log-probabilities as a `numpy` array (one value per branch), and combines with the current token log-prob:

```python
class SamplePowerDist:
    def __init__(
        self, alpha: float, n_branches: int, lookahead_depth: int,
        branch_sampler: BranchSampler,
    ) -> None: ...

    def __call__(self, context: GenerationContext) -> dict[str, float]:
        # For each candidate token t in context.token_probs:
        #   branch_log_probs = np.zeros(n_branches)
        #   For k in range(n_branches):
        #     self.branch_sampler.reset()
        #     branch_ctx = list(context.context_tokens) + context.tokenize_token(t)
        #     For d in range(lookahead_depth):
        #       next_lp = context.query_next(branch_ctx)
        #       if next_lp is None: break  # EOS — keep partial log_prob, no penalty
        #       next_token = self.branch_sampler.sample(next_lp)
        #       branch_log_probs[k] += log(prob_of_token(next_token, next_lp))
        #       branch_ctx += context.tokenize_token(next_token)
        #   new_log_prob(t) = alpha * context.token_probs[t] + f(branch_log_probs)
        #   where f() is the power-series combination over the array
        # Return {t: new_log_prob(t) for t in context.token_probs}
```

**Key design decisions captured here:**
- Per-branch log-probs kept as a `numpy` array of shape `(n_branches,)` per candidate, enabling power-series computations over them
- `branch_sampler.reset()` called at the start of each branch so stateful samplers (e.g. Metropolis) start fresh
- Early EOS terminates the branch loop; the partially accumulated `branch_log_prob` is used as-is with no penalty

### Step 7 — Update tests

- `gen_adj_model` fixture: `adjust_fn` mock now receives a `GenerationContext`; fixture must build and pass one
- `TestGenerateAdjusted`: `prev_probs` inspection tests read `call_args[0][0].prev_probs` (from context) instead of `call_args[0][1]`
- `TestSampleLowTempCall` and `TestAdjustIdentity`: updated to pass a `GenerationContext`
- New `TestBranchSampler`: covers `reset()` default no-op behaviour
- New `TestMetropolisSampler`: covers acceptance criterion, rejection, and state reset between branches
- New `TestSamplePowerDist`: covers constructor, return type/keys, `query_next` call count (verifies `n_branches × lookahead_depth` calls per candidate), early EOS handling (no penalty applied), and `branch_sampler.reset()` called once per branch

---

## Known Issues (`cli.py`, `ui_gen_data()`):
   - `data_size = int(input(...))` either succeeds (always an `int`) or raises an exception
   - The `while not isinstance(data_size, int)` guard below it can never be reached

2. **Hard-coded model parameters** (`llama_interface.py`):
   - `n_ctx=2048` and `max_tokens=512` are not configurable

3. **`FBT001` on `_LlamaInterface` protocol** (`analysis/tokenizer.py`):
   - `add_bos: bool` and `special: bool` are flagged as boolean positional arguments
   - These mirror the external `llama_cpp.Llama` API signature and cannot be made keyword-only

4. **`generate_adjusted()` returns `LLMOutputData`** — richer return type is a known limitation:
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
- ✓ `AdjustFn` type alias added (originally in `llama_interface.py`, moved to `adjust_probs.py`); signature updated to include `prev_probs: list[float]`
- ✓ `query_log_probs_next_token` signature updated: `context_tokens: list[int]` (was `list[str]`)
- ✓ `ModelInstance._format_chat_prompt()` implemented using `Jinja2ChatFormatter` (public API); previous attempts used non-existent `_chat_handler` then `_chat_handlers` dict — both wrong
- ✓ `ModelInstance.generate_adjusted()` implemented with `prev_probs` history, defensive copy passed to `adjust_fn`, and `None` guard for EOS detection
- ✓ `query_log_probs_next_token` crash on EOS fixed: empty `top_logprobs` list now returns `None` instead of raising `IndexError`
- ✓ `logprobs` parameter in `create_completion` corrected from `True` (bool) to `n_tokens` (int); `top_logprobs` parameter removed (not a `create_completion` parameter)
- ✓ Mutable `prev_probs` argument bug fixed: `list(prev_probs)` copy passed to `adjust_fn` so `MagicMock.call_args_list` records stable snapshots
- ✓ `Probabilities` class stub deleted (non-functional, multiple linter errors)
- ✓ `prob_of_token()` added to `analysis/probabilities.py`; exported from `analysis/__init__.py`
- ✓ `adjust_probs.py` module created: `AdjustFn`, `adjust_identity`, `SampleLowTemp`
- ✓ `ui_generate_adjusted()` and `get_adjusted_path()` added to `cli.py`; `logits_all=True` extended to `GENERATE_ADJUSTED` runs
- ✓ `raise UnexpectedFunctionError` → `raise UnexpectedFunctionError()` (instantiated correctly)
- ✓ Unit tests added: `test_adjust_probs.py`; `TestProbOfToken` in `test_probabilities.py`; new tests in `TestQueryLogProbsNextToken`, `TestFormatChatPrompt`, `TestGenerateAdjusted`
- ✓ Version bumped to 1.0.0
