"""Tests for Model in llama_interface.py."""

from contextlib import ExitStack
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from problm_solver.data import LLMNextTokenData, LLMOutputData, LLMOutputDataFull
from problm_solver.samplers import id_logprobs_to_candidate_tokens


def _make_llama_mock(response_text: str = 'Mock response.') -> MagicMock:
    """Return a MagicMock that mimics the low-level llama.cpp API used by Model."""
    mock_llm = MagicMock()
    mock_llm.metadata = {
        'general.architecture': 'llama',
        'llama.block_count': '32',
        'llama.attention.head_count_kv': '8',
        'llama.attention.head_count': '32',
        'llama.embedding_length': '4096',
    }
    mock_llm.n_ctx.return_value = 2048
    return mock_llm


@pytest.fixture
def model():
    """Return a Model with a mocked underlying Llama object."""
    from problm_solver.llama_interface import Model

    with patch('problm_solver.llama_interface.Llama') as MockLlama:
        MockLlama.return_value = _make_llama_mock('Test answer.')
        instance = Model(fname='fake.gguf', context='What is the answer?')
    return instance


class TestModelInit:
    """Tests for Model.__init__."""

    def test_logits_all_defaults_to_false(self) -> None:
        """logits_all defaults to False, so Llama is constructed without it set."""
        from problm_solver.llama_interface import Model

        with patch('problm_solver.llama_interface.Llama') as MockLlama:
            MockLlama.return_value = _make_llama_mock()
            Model(fname='fake.gguf', context='Hello')
            _, kwargs = MockLlama.call_args
            assert kwargs.get('logits_all') is False

    def test_logits_all_true_passed_to_llama(self) -> None:
        """logits_all=True is forwarded to the Llama constructor."""
        from problm_solver.llama_interface import Model

        with patch('problm_solver.llama_interface.Llama') as MockLlama:
            MockLlama.return_value = _make_llama_mock()
            Model(fname='fake.gguf', context='Hello', logits_all=True)
            _, kwargs = MockLlama.call_args
            assert kwargs.get('logits_all') is True

    def test_cache_sized_from_model_metadata(self) -> None:
        """LlamaRAMCache capacity is derived from model metadata and n_ctx."""
        from problm_solver.llama_interface import Model

        with patch('problm_solver.llama_interface.Llama') as MockLlama, \
             patch('problm_solver.llama_interface.LlamaRAMCache') as MockCache:
            MockLlama.return_value = _make_llama_mock()
            Model(fname='fake.gguf', context='Hello')
            _, kwargs = MockCache.call_args
            # 4 states × (2048 ctx × 2 K&V × 32 layers × 8 KV heads × 128 head_dim × 2 bytes)
            assert kwargs.get('capacity_bytes') == 4 * 2048 * 2 * 32 * 8 * 128 * 2

    def test_wires_c_backend(self) -> None:
        """Model always wires the ModelCBackend."""
        from problm_solver.llama_interface import Model
        from problm_solver.llama_lowlevel import ModelCBackend

        with patch('problm_solver.llama_interface.Llama') as mock_llama:
            mock_llama.return_value = _make_llama_mock()
            instance = Model(fname='fake.gguf', context='Hello')

        assert isinstance(instance._llm_backend, ModelCBackend)


class TestKVUnifiedContextShim:
    """Tests for the kv_unified injection shim used for multi-sequence decoding."""

    def test_shim_sets_kv_unified_true_within_context(self) -> None:
        """Inside the shim, the submodule binding Llama uses returns kv_unified=True."""
        from problm_solver.llama_interface import _kv_unified_default_params

        import llama_cpp

        # llama.py does `import llama_cpp.llama_cpp as llama_cpp`, so Llama reads
        # llama_context_default_params off the SUBMODULE. That binding must be patched.
        assert llama_cpp.llama_cpp.llama_context_default_params().kv_unified is False
        assert llama_cpp.llama_cpp.llama_context_default_params().n_seq_max in (0, 1)
        with _kv_unified_default_params(enabled=True):
            assert llama_cpp.llama_cpp.llama_context_default_params().kv_unified is True
            assert llama_cpp.llama_cpp.llama_context_default_params().n_seq_max == 64
        # Restored on exit.
        assert llama_cpp.llama_cpp.llama_context_default_params().kv_unified is False

    def test_shim_disabled_is_transparent(self) -> None:
        """With enabled=False the shim leaves the submodule binding untouched."""
        from problm_solver.llama_interface import _kv_unified_default_params

        import llama_cpp

        with _kv_unified_default_params(enabled=False):
            assert llama_cpp.llama_cpp.llama_context_default_params().kv_unified is False

    def test_model_requests_kv_unified_by_default(self) -> None:
        """Constructing Model enters the kv_unified shim by default."""
        from problm_solver.llama_interface import Model

        with patch('problm_solver.llama_interface.Llama') as MockLlama, \
             patch('problm_solver.llama_interface._kv_unified_default_params') as shim:
            MockLlama.return_value = _make_llama_mock()
            Model(fname='fake.gguf', context='Hello')
        # Default kv_unified=True → shim entered with enabled=True.
        shim.assert_called_once_with(enabled=True)

    def test_model_can_disable_kv_unified(self) -> None:
        """kv_unified=False skips the shim (stock llama.cpp context)."""
        from problm_solver.llama_interface import Model

        with patch('problm_solver.llama_interface.Llama') as MockLlama, \
             patch('problm_solver.llama_interface._kv_unified_default_params') as shim:
            MockLlama.return_value = _make_llama_mock()
            Model(fname='fake.gguf', context='Hello', kv_unified=False)
        shim.assert_called_once_with(enabled=False)


class TestModelQuery:
    """Tests for Model.query."""

@pytest.fixture
def low_level_model(model):
    """Extend model with low-level eval state for Phase 5 tests.

    - _format_chat_prompt returns [1, 2, 3] (3-token prompt)
    - vocab_size = 4, EOS = token 3
    - scores[2]: token 1 is argmax (non-EOS) — first generated token
    - scores[3]: token 3 is argmax (EOS)   — generation stops
    - detokenize([n], special=True)  -> b'tok<n>'
    - detokenize([n, ...], special=False) -> b'decoded output'
    """
    vocab_size = 4
    eos_id = 3
    scores = np.zeros((2048, vocab_size), dtype=np.float32)
    scores[2] = [0.0, 3.0, 1.0, -2.0]   # n_tokens=3 after prompt: argmax=1
    scores[3] = [-2.0, 0.0, 0.0, 5.0]   # n_tokens=4 after eval([1]): argmax=3=EOS

    model._llm.scores = scores
    model._llm.n_tokens = 0
    model._llm.token_eos.return_value = eos_id

    def mock_detokenize(ids, special=False):
        if special:
            return f'tok{ids[0]}'.encode()
        return b'decoded output'
    model._llm.detokenize.side_effect = mock_detokenize

    def mock_reset():
        model._llm.n_tokens = 0
    model._llm.reset.side_effect = mock_reset

    def mock_eval(tokens):
        model._llm.n_tokens += len(tokens)
    model._llm.eval.side_effect = mock_eval

    with patch.object(model, '_format_chat_prompt', return_value=[1, 2, 3]):
        yield model


class TestModelQuery:
    """Tests for Model.query."""

    def test_returns_string(self, low_level_model) -> None:
        """query() returns a plain string."""
        mock_rng = MagicMock()
        mock_rng.gumbel.return_value = np.zeros(4)
        with patch('problm_solver.llama_interface.resolve_rng', return_value=mock_rng):
            result = low_level_model.query()
        assert isinstance(result, str)

    def test_returns_detokenized_output(self, low_level_model) -> None:
        """query() returns the detokenized form of the generated token IDs."""
        mock_rng = MagicMock()
        mock_rng.gumbel.return_value = np.zeros(4)
        with patch('problm_solver.llama_interface.resolve_rng', return_value=mock_rng):
            result = low_level_model.query()
        assert result == 'decoded output'

    def test_calls_reset_then_eval_with_prompt(self, low_level_model) -> None:
        """query() calls reset() then eval() with the formatted prompt tokens."""
        mock_rng = MagicMock()
        mock_rng.gumbel.return_value = np.zeros(4)
        with patch('problm_solver.llama_interface.resolve_rng', return_value=mock_rng):
            low_level_model.query()
        low_level_model._llm.reset.assert_called_once()
        assert low_level_model._llm.eval.call_args_list[0] == call([1, 2, 3])

    def test_stops_at_eos(self, low_level_model) -> None:
        """query() stops and excludes EOS; only the token before EOS is in the output."""
        mock_rng = MagicMock()
        mock_rng.gumbel.return_value = np.zeros(4)
        with patch('problm_solver.llama_interface.resolve_rng', return_value=mock_rng):
            low_level_model.query()
        # eval: once for prompt, once for the non-EOS token; EOS is not eval'd
        assert low_level_model._llm.eval.call_count == 2


class TestModelQueryNTimes:
    """Tests for Model.query_n_times."""

    def test_returns_numpy_array(self, model) -> None:
        """query_n_times() returns a numpy array."""
        with patch.object(model, 'query', return_value='answer'):
            result = model.query_n_times(3)
        assert isinstance(result, np.ndarray)

    def test_array_length_matches_n(self, model) -> None:
        """query_n_times(n) returns exactly n responses."""
        with patch.object(model, 'query', return_value='answer'):
            result = model.query_n_times(5)
        assert len(result) == 5

    def test_query_called_n_times(self, model) -> None:
        """query_n_times(n) calls query() exactly n times."""
        with patch.object(model, 'query', return_value='answer') as mock_q:
            model.query_n_times(4)
        assert mock_q.call_count == 4

    def test_responses_are_strings(self, model) -> None:
        """All elements in the returned array are strings."""
        with patch.object(model, 'query', return_value='answer'):
            result = model.query_n_times(3)
        for item in result:
            assert isinstance(item, str)


class TestModelQueryLogProbs:
    """Tests for Model.query_log_probs."""

    def test_returns_llmtokendata(self, low_level_model) -> None:
        """query_log_probs() returns an LLMTokenData instance."""
        from problm_solver.data import LLMTokenData

        mock_rng = MagicMock()
        mock_rng.gumbel.return_value = np.zeros(4)
        with patch('problm_solver.llama_interface.resolve_rng', return_value=mock_rng):
            result = low_level_model.query_log_probs()
        assert isinstance(result, LLMTokenData)

    def test_tokens_are_strings(self, low_level_model) -> None:
        """Tokens in the returned LLMTokenData are decoded strings."""
        mock_rng = MagicMock()
        mock_rng.gumbel.return_value = np.zeros(4)
        with patch('problm_solver.llama_interface.resolve_rng', return_value=mock_rng):
            result = low_level_model.query_log_probs()
        assert all(isinstance(t, str) for t in result.tokens)

    def test_probs_are_exp_of_sampled_token_logprobs(self, low_level_model) -> None:
        """Each probability equals exp(log-prob) of the corresponding sampled token."""
        from problm_solver.llama_interface import Model

        mock_rng = MagicMock()
        mock_rng.gumbel.return_value = np.zeros(4)
        with patch('problm_solver.llama_interface.resolve_rng', return_value=mock_rng):
            result = low_level_model.query_log_probs()
        lp = Model._log_softmax(np.array([0.0, 3.0, 1.0, -2.0], dtype=np.float32))
        assert result.probs == pytest.approx([float(np.exp(lp[1]))])

    def test_probs_are_between_zero_and_one(self, low_level_model) -> None:
        """All probabilities are valid (in the range (0, 1])."""
        mock_rng = MagicMock()
        mock_rng.gumbel.return_value = np.zeros(4)
        with patch('problm_solver.llama_interface.resolve_rng', return_value=mock_rng):
            result = low_level_model.query_log_probs()
        assert all(0.0 < p <= 1.0 for p in result.probs)

    def test_tokens_and_probs_same_length(self, low_level_model) -> None:
        """Tokens and probs are positionally aligned and have equal length."""
        mock_rng = MagicMock()
        mock_rng.gumbel.return_value = np.zeros(4)
        with patch('problm_solver.llama_interface.resolve_rng', return_value=mock_rng):
            result = low_level_model.query_log_probs()
        assert len(result.tokens) == len(result.probs)

    def test_prompt_is_stored(self, low_level_model) -> None:
        """The prompt is stored on the returned LLMTokenData."""
        mock_rng = MagicMock()
        mock_rng.gumbel.return_value = np.zeros(4)
        with patch('problm_solver.llama_interface.resolve_rng', return_value=mock_rng):
            result = low_level_model.query_log_probs()
        assert result.prompt == 'What is the answer?'

    def test_stops_at_eos(self, low_level_model) -> None:
        """query_log_probs() accumulates only the tokens generated before EOS."""
        mock_rng = MagicMock()
        mock_rng.gumbel.return_value = np.zeros(4)
        with patch('problm_solver.llama_interface.resolve_rng', return_value=mock_rng):
            result = low_level_model.query_log_probs()
        # scores[2]=token1 (non-EOS), scores[3]=token3=EOS: exactly 1 token
        assert len(result.tokens) == 1


class TestModelGenerateData:
    """Tests for Model.generate_data."""

    def test_returns_llmoutputdata(self, model) -> None:
        """generate_data() returns an LLMOutputData instance."""
        with patch.object(model, 'query', return_value='answer'):
            result = model.generate_data(3)
        assert isinstance(result, LLMOutputData)

    def test_prompt_matches_context(self, model) -> None:
        """generate_data() stores the model's context string as the prompt."""
        with patch.object(model, 'query', return_value='answer'):
            result = model.generate_data(3)
        assert result.prompt == model.context

    def test_data_length_matches_n_samples(self, model) -> None:
        """generate_data(n) produces exactly n responses in the result."""
        with patch.object(model, 'query', return_value='answer'):
            result = model.generate_data(6)
        assert len(result.data) == 6

    def test_written_flag_is_false(self, model) -> None:
        """Freshly generated data has written=False — it hasn't been saved yet."""
        with patch.object(model, 'query', return_value='answer'):
            result = model.generate_data(2)
        assert result.written is False


class TestModelQueryBranch:
    """Tests for Model.query_branch."""

    # Vocabulary size and EOS token ID used across the fixture and tests.
    _VOCAB = 5
    _EOS = 4

    @pytest.fixture
    def branch_model(self, model):
        """Configure mock LLM for query_branch with full n_tokens state tracking.

        Context [10, 20, 30] has 3 tokens, so after reset + eval the first
        logit row queried is scores[2].  Each subsequent eval([token]) advances
        n_tokens by 1, so scores[3], scores[4] … are used in turn.

        save_state captures the current n_tokens value; load_state restores it,
        mirroring real llama_cpp behaviour.
        """
        scores = np.zeros((2048, self._VOCAB), dtype=np.float32)
        # scores[2]: token 1 is argmax (non-EOS)
        scores[2] = [0.5, 3.0, 1.5, 0.2, -2.0]
        # scores[3]: token 2 is argmax (non-EOS)
        scores[3] = [0.5, 0.5, 2.0, 0.2, -2.0]
        # scores[4]: token 0 is argmax (non-EOS)
        scores[4] = [2.0, 0.5, 0.5, 0.2, -2.0]

        model._llm.scores = scores
        model._llm.n_tokens = 0
        model._llm.token_eos.return_value = self._EOS

        n_tokens_snapshot = [0]
        saved_state = MagicMock()

        def mock_reset():
            model._llm.n_tokens = 0
        model._llm.reset.side_effect = mock_reset

        def mock_eval(tokens):
            model._llm.n_tokens += len(tokens)
        model._llm.eval.side_effect = mock_eval

        def mock_save_state():
            n_tokens_snapshot[0] = model._llm.n_tokens
            return saved_state
        model._llm.save_state.side_effect = mock_save_state

        def mock_load_state(state):
            if state is saved_state:
                model._llm.n_tokens = n_tokens_snapshot[0]
        model._llm.load_state.side_effect = mock_load_state

        # Expose the sentinel so tests can assert the exact object passed.
        model._test_saved_state = saved_state
        return model

    def test_returns_float(self, branch_model) -> None:
        """query_branch() returns a Python float."""
        mock_rng = MagicMock()
        mock_rng.gumbel.return_value = np.zeros(self._VOCAB)
        with patch('problm_solver.llama_interface.resolve_rng', return_value=mock_rng):
            result = branch_model.query_branch([10, 20, 30], max_tokens=1)
        assert isinstance(result, float)

    def test_returns_zero_for_immediate_eos(self, branch_model) -> None:
        """Returns 0.0 when the first sampled token is EOS."""
        # Make EOS the argmax by giving it an overwhelming logit.
        branch_model._llm.scores[2] = [-10.0, -10.0, -10.0, -10.0, 10.0]
        mock_rng = MagicMock()
        mock_rng.gumbel.return_value = np.zeros(self._VOCAB)
        with patch('problm_solver.llama_interface.resolve_rng', return_value=mock_rng):
            result = branch_model.query_branch([10, 20, 30], max_tokens=5)
        assert result == 0.0

    def test_sums_log_probs_of_generated_tokens(self, branch_model) -> None:
        """Return value equals the sum of log-probs of the sampled tokens."""
        from problm_solver.llama_interface import Model

        # With Gumbel noise = 0, sampling is greedy: argmax of logprobs.
        # Step 1: scores[2], argmax = 1 (logit 3.0)
        # Step 2: scores[3], argmax = 2 (logit 2.0)
        lp1 = float(Model._log_softmax(
            np.array([0.5, 3.0, 1.5, 0.2, -2.0], dtype=np.float32)
        )[1])
        lp2 = float(Model._log_softmax(
            np.array([0.5, 0.5, 2.0, 0.2, -2.0], dtype=np.float32)
        )[2])
        mock_rng = MagicMock()
        mock_rng.gumbel.return_value = np.zeros(self._VOCAB)
        with patch('problm_solver.llama_interface.resolve_rng', return_value=mock_rng):
            result = branch_model.query_branch([10, 20, 30], max_tokens=2)
        assert result == pytest.approx(lp1 + lp2)

    def test_stops_at_max_tokens(self, branch_model) -> None:
        """Generation stops after exactly max_tokens tokens when EOS never appears."""
        mock_rng = MagicMock()
        mock_rng.gumbel.return_value = np.zeros(self._VOCAB)
        with patch('problm_solver.llama_interface.resolve_rng', return_value=mock_rng):
            branch_model.query_branch([10, 20, 30], max_tokens=2)
        # eval: once for context, once per generated token (2 tokens)
        assert branch_model._llm.eval.call_count == 3

    def test_eos_log_prob_not_included_in_sum(self, branch_model) -> None:
        """The log-probability of the EOS token itself is not added to the total."""
        from problm_solver.llama_interface import Model

        # Step 1 generates token 1 (non-EOS); step 2 generates EOS.
        branch_model._llm.scores[3] = [-10.0, -10.0, -10.0, -10.0, 10.0]  # EOS argmax
        lp1 = float(Model._log_softmax(
            np.array([0.5, 3.0, 1.5, 0.2, -2.0], dtype=np.float32)
        )[1])
        mock_rng = MagicMock()
        mock_rng.gumbel.return_value = np.zeros(self._VOCAB)
        with patch('problm_solver.llama_interface.resolve_rng', return_value=mock_rng):
            result = branch_model.query_branch([10, 20, 30], max_tokens=5)
        assert result == pytest.approx(lp1)

    def test_calls_reset(self, branch_model) -> None:
        """reset() is called once to clear stale KV-cache state."""
        mock_rng = MagicMock()
        mock_rng.gumbel.return_value = np.zeros(self._VOCAB)
        with patch('problm_solver.llama_interface.resolve_rng', return_value=mock_rng):
            branch_model.query_branch([10, 20, 30], max_tokens=1)
        branch_model._llm.reset.assert_called_once()

    def test_calls_eval_with_context_tokens(self, branch_model) -> None:
        """eval() is first called with the full context token list."""
        context = [10, 20, 30]
        mock_rng = MagicMock()
        mock_rng.gumbel.return_value = np.zeros(self._VOCAB)
        with patch('problm_solver.llama_interface.resolve_rng', return_value=mock_rng):
            branch_model.query_branch(context, max_tokens=1)
        assert branch_model._llm.eval.call_args_list[0] == call(context)

    def test_saves_state_after_context_eval(self, branch_model) -> None:
        """save_state() is called exactly once, after evaluating the context."""
        mock_rng = MagicMock()
        mock_rng.gumbel.return_value = np.zeros(self._VOCAB)
        with patch('problm_solver.llama_interface.resolve_rng', return_value=mock_rng):
            branch_model.query_branch([10, 20, 30], max_tokens=1)
        branch_model._llm.save_state.assert_called_once()
        # The n_tokens captured at save time equals len(context_tokens).
        assert branch_model._llm.n_tokens >= 3  # at least context + 1 generated

    def test_loads_saved_state(self, branch_model) -> None:
        """load_state() is called with exactly the object returned by save_state()."""
        mock_rng = MagicMock()
        mock_rng.gumbel.return_value = np.zeros(self._VOCAB)
        with patch('problm_solver.llama_interface.resolve_rng', return_value=mock_rng):
            branch_model.query_branch([10, 20, 30], max_tokens=1)
        branch_model._llm.load_state.assert_called_once_with(
            branch_model._test_saved_state
        )

    def test_eval_called_once_per_generated_token(self, branch_model) -> None:
        """eval() is called once for the context and once for each generated token."""
        mock_rng = MagicMock()
        mock_rng.gumbel.return_value = np.zeros(self._VOCAB)
        with patch('problm_solver.llama_interface.resolve_rng', return_value=mock_rng):
            branch_model.query_branch([10, 20, 30], max_tokens=3)
        # 1 context eval + 3 single-token evals
        assert branch_model._llm.eval.call_count == 4


class TestModelQueryBranchesFromLiveBatch:
    """Tests for Model.query_branches_from_live_batch.

    The batched path generates ``n_branches`` independent branches in lockstep
    from the currently loaded live (seq-0) state, using multi-sequence
    batched decoding. With zero-Gumbel sampling every branch greedily picks
    the argmax of the per-position scores row, so all branches produce the
    same log-probability as a single sequential ``query_branch_from_live``
    call — that is the core equivalence invariant.
    """

    _VOCAB = 5
    _EOS = 4
    _ROOT_POS = 3  # live seq-0 length; first logits read from scores[ROOT_POS - 1]

    @pytest.fixture
    def batch_model(self, model):
        """Configure mock LLM with per-position scores for batched branch tests.

        The live state is positioned at ``_ROOT_POS`` (``n_tokens = 3``), so
        :meth:`last_logits` returns ``scores[2]``. ``decode_batch``'s mock
        fallback returns ``scores[positions]``, so step ``s`` consumes
        ``scores[_ROOT_POS - 1 + s]`` — mirroring the sequential path.
        """
        scores = np.zeros((2048, self._VOCAB), dtype=np.float32)
        # step 0 (position 2): argmax = 1 (non-EOS)
        scores[2] = [0.5, 3.0, 1.5, 0.2, -2.0]
        # step 1 (position 3): argmax = 2 (non-EOS)
        scores[3] = [0.5, 0.5, 2.0, 0.2, -2.0]
        # step 2 (position 4): argmax = 0 (non-EOS) — only used if depth > 2
        scores[4] = [2.0, 0.5, 0.5, 0.2, -2.0]

        model._llm.scores = scores
        model._llm.n_tokens = self._ROOT_POS
        model._llm.token_eos.return_value = self._EOS
        # decode_batch's mock fallback calls llm.eval; leave it as a no-op mock
        # (n_tokens advancement is irrelevant — batched logits come from
        # scores[positions], not scores[n_tokens - 1]).
        return model

    def _zero_gumbel_rng(self) -> MagicMock:
        """Return an rng mock whose ``gumbel`` returns zeros of vocab width."""
        mock_rng = MagicMock()
        mock_rng.gumbel.return_value = np.zeros(self._VOCAB, dtype=np.float64)
        return mock_rng

    def test_returns_array_of_n_branches_floats(self, batch_model) -> None:
        """Return shape is (n_branches,) float64."""
        with patch('problm_solver.llama_interface.resolve_rng',
                   return_value=self._zero_gumbel_rng()):
            result = batch_model.query_branches_from_live_batch(
                max_tokens=2, n_branches=3
            )
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert result.dtype == np.float64

    def test_zero_branches_returns_empty_array(self, batch_model) -> None:
        """n_branches=0 returns an empty (0,) array without touching the KV cache."""
        with patch('problm_solver.llama_interface.resolve_rng',
                   return_value=self._zero_gumbel_rng()):
            result = batch_model.query_branches_from_live_batch(
                max_tokens=2, n_branches=0
            )
        assert result.shape == (0,)
        batch_model._llm._ctx.kv_cache_seq_cp.assert_not_called()  # noqa: SLF001

    def test_immediate_eos_branch_logprob_is_zero(self, batch_model) -> None:
        """A branch whose first sampled token is EOS gets log-prob 0.0."""
        batch_model._llm.scores[2] = [-10.0, -10.0, -10.0, -10.0, 10.0]  # EOS argmax
        with patch('problm_solver.llama_interface.resolve_rng',
                   return_value=self._zero_gumbel_rng()):
            result = batch_model.query_branches_from_live_batch(
                max_tokens=5, n_branches=3
            )
        assert np.allclose(result, 0.0)

    def test_sums_log_probs_of_generated_tokens(self, batch_model) -> None:
        """Each branch's log-prob equals the sum of per-step sampled log-probs."""
        from problm_solver.llama_interface import Model

        lp1 = float(Model._log_softmax(
            np.array([0.5, 3.0, 1.5, 0.2, -2.0], dtype=np.float32)
        )[1])
        lp2 = float(Model._log_softmax(
            np.array([0.5, 0.5, 2.0, 0.2, -2.0], dtype=np.float32)
        )[2])
        with patch('problm_solver.llama_interface.resolve_rng',
                   return_value=self._zero_gumbel_rng()):
            result = batch_model.query_branches_from_live_batch(
                max_tokens=2, n_branches=4
            )
        assert np.allclose(result, lp1 + lp2)

    def test_all_branches_equal_under_deterministic_sampling(self, batch_model) -> None:
        """With zero Gumbel, every branch follows the same greedy path."""
        with patch('problm_solver.llama_interface.resolve_rng',
                   return_value=self._zero_gumbel_rng()):
            result = batch_model.query_branches_from_live_batch(
                max_tokens=3, n_branches=5
            )
        assert np.allclose(result, result[0])

    def test_stops_at_max_tokens_without_eos(self, batch_model) -> None:
        """A full-depth run with no EOS sums exactly max_tokens log-probs."""
        from problm_solver.llama_interface import Model

        lp1 = float(Model._log_softmax(
            np.array([0.5, 3.0, 1.5, 0.2, -2.0], dtype=np.float32)
        )[1])
        lp2 = float(Model._log_softmax(
            np.array([0.5, 0.5, 2.0, 0.2, -2.0], dtype=np.float32)
        )[2])
        lp3 = float(Model._log_softmax(
            np.array([2.0, 0.5, 0.5, 0.2, -2.0], dtype=np.float32)
        )[0])
        with patch('problm_solver.llama_interface.resolve_rng',
                   return_value=self._zero_gumbel_rng()):
            result = batch_model.query_branches_from_live_batch(
                max_tokens=3, n_branches=2
            )
        assert np.allclose(result, lp1 + lp2 + lp3)

    def test_mid_branch_eos_stops_that_branch_only(self, batch_model) -> None:
        """EOS at step 1 freezes that branch; others continue to step 2.

        Because every branch shares per-position scores under zero Gumbel,
        EOS at step 1 affects all branches simultaneously — so this test
        instead asserts the accumulated value is just step-0's log-prob when
        step 1 is EOS.
        """
        from problm_solver.llama_interface import Model

        # step 1 (position 3) argmax = EOS
        batch_model._llm.scores[3] = [-10.0, -10.0, -10.0, -10.0, 10.0]
        lp1 = float(Model._log_softmax(
            np.array([0.5, 3.0, 1.5, 0.2, -2.0], dtype=np.float32)
        )[1])
        with patch('problm_solver.llama_interface.resolve_rng',
                   return_value=self._zero_gumbel_rng()):
            result = batch_model.query_branches_from_live_batch(
                max_tokens=5, n_branches=3
            )
        assert np.allclose(result, lp1)

    def test_clones_root_kv_into_each_branch_sequence(self, batch_model) -> None:
        """kv_cache_seq_cp is called once per branch, copying seq 0 → branch id."""
        with patch('problm_solver.llama_interface.resolve_rng',
                   return_value=self._zero_gumbel_rng()):
            batch_model.query_branches_from_live_batch(max_tokens=2, n_branches=3)
        cp_calls = batch_model._llm._ctx.kv_cache_seq_cp.call_args_list  # noqa: SLF001
        assert len(cp_calls) == 3
        # Each call clones sequence 0 into a distinct branch id over [0, ROOT_POS).
        dest_ids = {c.args[1] for c in cp_calls}
        assert dest_ids == {1, 2, 3}
        for c in cp_calls:
            assert c.args[0] == 0
            assert c.args[2] == 0
            assert c.args[3] == self._ROOT_POS

    def test_tears_down_branch_sequences_on_cleanup(self, batch_model) -> None:
        """After scoring, only seq 0 is retained (kv_cache_seq_keep(0))."""
        with patch('problm_solver.llama_interface.resolve_rng',
                   return_value=self._zero_gumbel_rng()):
            batch_model.query_branches_from_live_batch(max_tokens=2, n_branches=3)
        batch_model._llm._ctx.kv_cache_seq_keep.assert_called_once_with(0)  # noqa: SLF001

    def test_does_not_mutate_model_n_tokens(self, batch_model) -> None:
        """Batched branch generation leaves the live seq-0 token count unchanged."""
        original = batch_model._llm_backend.n_tokens
        with patch('problm_solver.llama_interface.resolve_rng',
                   return_value=self._zero_gumbel_rng()):
            batch_model.query_branches_from_live_batch(max_tokens=3, n_branches=4)
        assert batch_model._llm_backend.n_tokens == original

    def test_batched_decode_call_count_is_depth_minus_one(self, batch_model) -> None:
        """Only ``depth - 1`` batched decodes are issued (final step need not decode)."""
        with patch('problm_solver.llama_interface.resolve_rng',
                   return_value=self._zero_gumbel_rng()):
            batch_model.query_branches_from_live_batch(max_tokens=3, n_branches=2)
        # depth=3 → decodes after step 0 and step 1 only (step 2 is terminal).
        assert batch_model._llm.eval.call_count == 2

    def test_end_to_end_equivalent_to_sequential_query_branch_from_live(
        self, batch_model
    ) -> None:
        """Every batched branch equals a sequential query_branch_from_live call.

        Under zero-Gumbel sampling both paths greedily pick the argmax of the
        same per-position scores row, so each batched branch's total log-prob
        must equal the sequential branch's total. This is the authoritative
        proof that batching introduces no semantic drift.
        """
        # The sequential path reads scores[n_tokens - 1] and relies on eval()
        # advancing n_tokens to the next row; wire that up here. The batched
        # path reads scores[positions] directly and is unaffected by the drift.
        def mock_eval(tokens):
            batch_model._llm.n_tokens += len(tokens)
        batch_model._llm.eval.side_effect = mock_eval

        with patch('problm_solver.llama_interface.resolve_rng',
                   return_value=self._zero_gumbel_rng()):
            batched = batch_model.query_branches_from_live_batch(
                max_tokens=2, n_branches=3
            )
        # Restore the live position before running the sequential branch.
        batch_model._llm.n_tokens = self._ROOT_POS
        with patch('problm_solver.llama_interface.resolve_rng',
                   return_value=self._zero_gumbel_rng()):
            sequential = batch_model.query_branch_from_live(max_tokens=2)

        assert np.allclose(batched, sequential)


class TestQueryLogProbsNextToken:
    """Tests for Model.query_log_probs_next_token."""

    @pytest.fixture
    def next_token_model(self, model):
        """Configure mock LLM for query_log_probs_next_token.

        scores[n_tokens - 1] is the logit row used, so after reset() +
        eval([1, 2, 3]) n_tokens=3 and the test values sit at scores[2].
        """
        vocab_size = 5
        scores = np.zeros((2048, vocab_size), dtype=np.float32)
        scores[2] = [0.0, 3.0, 1.0, 2.0, 0.5]  # token 1 highest, token 3 second
        model._llm.scores = scores
        model._llm.n_tokens = 0
        model._llm.detokenize.side_effect = (
            lambda ids, special=False: f'<tok{ids[0]}>'.encode()
        )

        def mock_reset():
            model._llm.n_tokens = 0
        model._llm.reset.side_effect = mock_reset

        def mock_eval(tokens):
            model._llm.n_tokens += len(tokens)
        model._llm.eval.side_effect = mock_eval

        return model

    def test_always_returns_llmnexttokendata(self, next_token_model) -> None:
        """query_log_probs_next_token() always returns LLMNextTokenData, never None."""
        result = next_token_model.query_log_probs_next_token([1, 2, 3], n_tokens=2)
        assert isinstance(result, LLMNextTokenData)
        assert result is not None

    def test_output_vec_is_passed_context(self, next_token_model) -> None:
        """output_vec on the result is the context list that was passed in."""
        context = [1, 2, 3]
        result = next_token_model.query_log_probs_next_token(context, n_tokens=2)
        assert result.output_vec == context

    def test_top_k_tokens_contains_highest_scoring_tokens(self, next_token_model) -> None:
        """top_k_tokens contains the n tokens with the highest logits."""
        result = next_token_model.query_log_probs_next_token([1, 2, 3], n_tokens=2)
        assert '<tok1>' in result.top_k_tokens
        assert '<tok3>' in result.top_k_tokens

    def test_top_k_tokens_has_n_entries(self, next_token_model) -> None:
        """top_k_tokens contains exactly n_tokens entries."""
        result = next_token_model.query_log_probs_next_token([1, 2, 3], n_tokens=3)
        assert len(result.top_k_tokens) == 3

    def test_calls_reset(self, next_token_model) -> None:
        """reset() is called once to clear stale KV-cache state."""
        next_token_model.query_log_probs_next_token([1, 2, 3], n_tokens=2)
        next_token_model._llm.reset.assert_called_once()

    def test_calls_eval_with_context_tokens(self, next_token_model) -> None:
        """eval() is called with the full context token list."""
        context = [10, 20, 30]
        next_token_model.query_log_probs_next_token(context, n_tokens=2)
        next_token_model._llm.eval.assert_called_once_with(context)


class TestFormatChatPrompt:
    """Tests for Model._format_chat_prompt."""

    @pytest.fixture
    def chat_prompt_model(self, model):
        """Configure the mock LLM for _format_chat_prompt."""
        mock_result = MagicMock()
        mock_result.prompt = 'formatted prompt string'
        model._llm.metadata = {'tokenizer.chat_template': 'dummy_template'}
        model._llm.token_eos.return_value = 2
        model._llm.token_bos.return_value = 1
        model._llm.detokenize.return_value = b''
        model._llm.tokenize.return_value = [1, 2, 3, 4, 5]
        with patch(
            'problm_solver.llama_interface.Jinja2ChatFormatter'
        ) as mock_jinja:
            mock_jinja.return_value.return_value = mock_result
            yield model

    def test_returns_list(self, chat_prompt_model) -> None:
        """_format_chat_prompt() returns a list."""
        result = chat_prompt_model._format_chat_prompt()
        assert isinstance(result, list)

    def test_returns_list_of_ints(self, chat_prompt_model) -> None:
        """All elements in the returned list are integers (token IDs)."""
        result = chat_prompt_model._format_chat_prompt()
        assert all(isinstance(x, int) for x in result)

    def test_formatter_constructed_with_metadata_template(self, chat_prompt_model) -> None:
        """Jinja2ChatFormatter is constructed using the template from model metadata."""
        with patch('problm_solver.llama_interface.Jinja2ChatFormatter') as mock_jinja:
            mock_result = MagicMock()
            mock_result.prompt = 'p'
            mock_jinja.return_value.return_value = mock_result
            chat_prompt_model._format_chat_prompt()
            args, kwargs = mock_jinja.call_args
            assert kwargs.get('template') == 'dummy_template'

    def test_formatter_called_with_user_message(self, chat_prompt_model) -> None:
        """The formatter instance is called with the context as a user-role message."""
        with patch('problm_solver.llama_interface.Jinja2ChatFormatter') as mock_jinja:
            mock_result = MagicMock()
            mock_result.prompt = 'p'
            mock_jinja.return_value.return_value = mock_result
            chat_prompt_model._format_chat_prompt()
            _, kwargs = mock_jinja.return_value.call_args
            assert kwargs.get('messages') == [
                {'role': 'user', 'content': chat_prompt_model.context}
            ]

    def test_calls_tokenize_with_handler_output(self, chat_prompt_model) -> None:
        """Tokenize is called on the encoded string returned by the chat handler."""
        chat_prompt_model._format_chat_prompt()
        chat_prompt_model._llm.tokenize.assert_called_once_with(
            b'formatted prompt string',
            add_bos=False,
            special=True,
        )

    def test_returns_tokenize_output(self, chat_prompt_model) -> None:
        """The return value is whatever tokenize() returns."""
        result = chat_prompt_model._format_chat_prompt()
        assert result == [1, 2, 3, 4, 5]


class TestLogSoftmax:
    """Tests for Model._log_softmax."""

    def test_output_is_valid_log_probability_distribution(self) -> None:
        """exp(log_softmax(x)) sums to 1.0 over the full vocabulary."""
        from problm_solver.llama_interface import Model

        logits = np.array([1.0, 2.0, 0.5, -1.0], dtype=np.float32)
        result = Model._log_softmax(logits)
        assert np.exp(result).sum() == pytest.approx(1.0)

    def test_argmax_is_preserved(self) -> None:
        """The token with the highest logit has the highest log-probability."""
        from problm_solver.llama_interface import Model

        logits = np.array([0.1, 3.0, -1.0, 0.5], dtype=np.float32)
        result = Model._log_softmax(logits)
        assert np.argmax(result) == np.argmax(logits)

    def test_all_values_are_non_positive(self) -> None:
        """All log-probabilities are ≤ 0 (probabilities are in (0, 1])."""
        from problm_solver.llama_interface import Model

        logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = Model._log_softmax(logits)
        assert np.all(result <= 0.0)

    def test_returns_float64_array(self) -> None:
        """Output dtype is float64 regardless of the input dtype."""
        from problm_solver.llama_interface import Model

        logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = Model._log_softmax(logits)
        assert result.dtype == np.float64

    def test_output_shape_matches_input(self) -> None:
        """Output array has the same shape as the input logits."""
        from problm_solver.llama_interface import Model

        logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        result = Model._log_softmax(logits)
        assert result.shape == logits.shape

    def test_numerically_stable_with_large_logits(self) -> None:
        """Does not overflow when logits are in the hundreds, as is common for LLMs."""
        from problm_solver.llama_interface import Model

        logits = np.array([300.0, 200.0, 100.0], dtype=np.float32)
        result = Model._log_softmax(logits)
        assert np.all(np.isfinite(result))
        assert np.exp(result).sum() == pytest.approx(1.0)

    def test_uniform_logits_produce_equal_log_probs(self) -> None:
        """All-equal logits map to the same log-probability for every token."""
        from problm_solver.llama_interface import Model

        logits = np.full(5, 2.0, dtype=np.float32)
        result = Model._log_softmax(logits)
        assert np.allclose(result, result[0])


class TestTopKIdsFromLogprobs:
    """Tests for Model._top_k_ids_from_logprobs."""

    def test_returns_exactly_n_entries(self, model) -> None:
        """The returned list has exactly n entries."""
        logprobs = np.array([-3.0, -1.0, -0.5, -2.0, -4.0], dtype=np.float64)
        result = model._top_k_ids_from_logprobs(logprobs, n=3)
        assert len(result) == 3

    def test_contains_highest_logprob_tokens(self, model) -> None:
        """Result contains the n token IDs with the highest log-probabilities."""
        logprobs = np.array([-3.0, -1.0, -0.5, -2.0, -4.0], dtype=np.float64)
        result = model._top_k_ids_from_logprobs(logprobs, n=2)
        ids = [idx for idx, _ in result]
        assert ids == [2, 1]

    def test_excludes_lower_logprob_tokens(self, model) -> None:
        """Token IDs outside the top-n are not present in the result."""
        logprobs = np.array([-3.0, -1.0, -0.5, -2.0, -4.0], dtype=np.float64)
        result = model._top_k_ids_from_logprobs(logprobs, n=2)
        ids = {idx for idx, _ in result}
        assert ids == {1, 2}

    def test_values_match_logprobs_of_their_tokens(self, model) -> None:
        """Each tuple value equals the log-probability at the corresponding vocab index."""
        logprobs = np.array([-3.0, -1.0, -0.5, -2.0, -4.0], dtype=np.float64)
        result = model._top_k_ids_from_logprobs(logprobs, n=3)
        assert [idx for idx, _ in result] == [2, 1, 3]
        assert result[0][1] == pytest.approx(-0.5)
        assert result[1][1] == pytest.approx(-1.0)
        assert result[2][1] == pytest.approx(-2.0)

    def test_values_are_python_floats(self, model) -> None:
        """All returned log-probs are plain Python floats, not numpy scalars."""
        logprobs = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
        result = model._top_k_ids_from_logprobs(logprobs, n=2)
        assert all(type(v) is float for _, v in result)

    def test_sorted_descending_by_log_prob(self, model) -> None:
        """Entries are ordered from highest to lowest log-probability."""
        logprobs = np.array([-3.0, -1.0, -0.5, -2.0, -4.0], dtype=np.float64)
        result = model._top_k_ids_from_logprobs(logprobs, n=3)
        values = [v for _, v in result]
        assert values == sorted(values, reverse=True)

    def test_n_clamped_to_vocab_size(self, model) -> None:
        """Requesting more tokens than vocab size returns every token."""
        logprobs = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
        result = model._top_k_ids_from_logprobs(logprobs, n=100)
        assert len(result) == 3

    def test_n_of_one_returns_single_highest_token(self, model) -> None:
        """n=1 returns only the argmax token with its log-probability."""
        logprobs = np.array([-3.0, -0.1, -2.0], dtype=np.float64)
        result = model._top_k_ids_from_logprobs(logprobs, n=1)
        assert len(result) == 1
        assert result[0][0] == 1
        assert result[0][1] == pytest.approx(-0.1)


@pytest.fixture
def gen_smpl_model(model):
    """Model with all generate_with_sampler() dependencies mocked.

    - _format_chat_prompt returns [10, 20, 30] (prompt_length = 3)
    - vocab_size = 4; EOS = token 0
    - scores[2] = [−10, 3, 1, 0.5]: top-2 are '<tok1>' and '<tok2>'
    - sample_from_logprobs (in llama_interface module) returns ' hello'
    - _llm.tokenize() returns [42] (a non-EOS token ID)
    - _llm.detokenize([tid], special=True) returns b'<tokN>'
    - reset/eval side-effects maintain n_tokens
    """
    vocab_size = 4
    scores = np.zeros((2048, vocab_size), dtype=np.float32)
    # n_tokens=3 after prompt eval; same logits for all subsequent positions
    for pos in range(2, 10):
        scores[pos] = [-10.0, 3.0, 1.0, 0.5]

    model._llm.scores = scores
    model._llm.n_tokens = 0
    model._logits_all = True
    model._llm.token_eos.return_value = 0
    model._llm.tokenize.return_value = [42]
    model._llm.detokenize.side_effect = (
        lambda ids, special=False: f'<tok{ids[0]}>'.encode()
    )

    def mock_reset():
        model._llm.n_tokens = 0
    model._llm.reset.side_effect = mock_reset

    def mock_eval(tokens):
        model._llm.n_tokens += len(tokens)
    model._llm.eval.side_effect = mock_eval

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(model, '_format_chat_prompt', return_value=[10, 20, 30])
        )
        stack.enter_context(
            patch('problm_solver.llama_interface.sample_from_logprobs', return_value=' hello')
        )
        stack.enter_context(
            patch('problm_solver.llama_interface.prob_of_token', return_value=0.8)
        )
        yield model


class TestGenerateWithSampler:
    """Tests for Model.generate_with_sampler."""

    def test_returns_llmoutputdatafull(self, gen_smpl_model) -> None:
        """generate_with_sampler() returns an LLMOutputDataFull instance."""
        result = gen_smpl_model.generate_with_sampler(top_k=2, top_p=1.0, adjust_fn=lambda ctx: ctx.token_id_probs, max_tokens=3)
        assert isinstance(result, LLMOutputDataFull)

    def test_context_is_list_of_strings(self, gen_smpl_model) -> None:
        """Context on the returned LLMOutputDataFull is a list of strings."""
        result = gen_smpl_model.generate_with_sampler(top_k=2, top_p=1.0, adjust_fn=lambda ctx: ctx.token_id_probs, max_tokens=3)
        assert isinstance(result.context, list)
        assert all(isinstance(s, str) for s in result.context)

    def test_written_flag_is_false(self, gen_smpl_model) -> None:
        """Freshly generated data has _written=False."""
        result = gen_smpl_model.generate_with_sampler(top_k=2, top_p=1.0, adjust_fn=lambda ctx: ctx.token_id_probs, max_tokens=3)
        assert result._written is False

    def test_loops_exactly_max_tokens_times(self, gen_smpl_model) -> None:
        """eval() is called once for the prompt then once per generated token."""
        gen_smpl_model.generate_with_sampler(top_k=2, top_p=1.0, adjust_fn=lambda ctx: ctx.token_id_probs, max_tokens=4)
        # 1 prompt eval + 4 token evals
        assert gen_smpl_model._llm.eval.call_count == 5

    def test_adjust_fn_called_each_step(self, gen_smpl_model) -> None:
        """adjust_fn is called once per generated token."""
        adjust_fn = MagicMock(return_value=id_logprobs_to_candidate_tokens({1: -0.5}))
        gen_smpl_model.generate_with_sampler(top_k=2, top_p=1.0, adjust_fn=adjust_fn, max_tokens=3)
        assert adjust_fn.call_count == 3

    def test_adjust_fn_receives_top_k_tokens(self, gen_smpl_model) -> None:
        """adjust_fn receives a GenerationContext whose token_probs is built from scores."""
        from problm_solver.llama_interface import Model

        adjust_fn = MagicMock(return_value=id_logprobs_to_candidate_tokens({1: -0.5}))
        gen_smpl_model.generate_with_sampler(top_k=2, top_p=1.0, adjust_fn=adjust_fn, max_tokens=1)
        ctx = adjust_fn.call_args[0][0]
        lp = Model._log_softmax(np.array([-10.0, 3.0, 1.0, 0.5], dtype=np.float32))
        assert ctx.token_id_probs.candidate_ids.tolist() == [1, 2]
        assert ctx.token_id_probs.candidate_logprobs.tolist() == pytest.approx([float(lp[1]), float(lp[2])])

    def test_adjust_fn_receives_empty_prev_probs_on_first_step(self, gen_smpl_model) -> None:
        """adjust_fn receives a GenerationContext with empty prev_probs on the first step."""
        adjust_fn = MagicMock(return_value=id_logprobs_to_candidate_tokens({1: -0.5}))
        gen_smpl_model.generate_with_sampler(top_k=2, top_p=1.0, adjust_fn=adjust_fn, max_tokens=1)
        ctx = adjust_fn.call_args_list[0][0][0]
        assert ctx.prev_probs == []

    def test_adjust_fn_receives_growing_prev_probs(self, gen_smpl_model) -> None:
        """prev_probs grows by one entry per step, containing prob_of_token return values."""
        adjust_fn = MagicMock(return_value=id_logprobs_to_candidate_tokens({1: -0.5}))
        gen_smpl_model.generate_with_sampler(top_k=2, top_p=1.0, adjust_fn=adjust_fn, max_tokens=3)
        # Step 1: prev_probs = []; then deterministic single-candidate prob=1.0
        assert adjust_fn.call_args_list[0][0][0].prev_probs == []
        assert adjust_fn.call_args_list[1][0][0].prev_probs == [1.0]
        assert adjust_fn.call_args_list[2][0][0].prev_probs == [1.0, 1.0]

    def test_response_topk_tokens_are_sampled_tokens(self, gen_smpl_model) -> None:
        """response_topk[0] contains the token strings chosen at each step."""
        result = gen_smpl_model.generate_with_sampler(top_k=2, top_p=1.0, adjust_fn=lambda ctx: ctx.token_id_probs, max_tokens=2)
        assert len(result.response_topk[0]) == 2

    def test_stops_early_on_eos_token(self, gen_smpl_model) -> None:
        """The loop breaks before max_tokens when tokenize returns the EOS token ID."""
        gen_smpl_model._llm.scores[2] = [3.0, 1.0, 0.0, -1.0]
        gen_smpl_model.generate_with_sampler(top_k=1, top_p=1.0, adjust_fn=lambda ctx: ctx.token_id_probs, max_tokens=10)
        # Only the prompt eval ran; no token eval because first sample was EOS
        assert gen_smpl_model._llm.eval.call_count == 1

    def test_stops_early_on_eos_argmax(self, gen_smpl_model) -> None:
        """The loop breaks when EOS is deterministically selected."""
        gen_smpl_model._llm.scores[2] = [3.0, 1.0, 0.0, -1.0]
        gen_smpl_model.generate_with_sampler(top_k=1, top_p=1.0, adjust_fn=lambda ctx: ctx.token_id_probs, max_tokens=10)
        assert gen_smpl_model._llm.eval.call_count == 1

    def test_prev_probs_reset_between_calls(self, gen_smpl_model) -> None:
        """prev_probs starts empty on every call to generate_with_sampler, not carried over."""
        adjust_fn = MagicMock(return_value=id_logprobs_to_candidate_tokens({1: -0.5}))
        gen_smpl_model.generate_with_sampler(top_k=2, top_p=1.0, adjust_fn=adjust_fn, max_tokens=2)
        gen_smpl_model._llm.eval.reset_mock()
        adjust_fn.reset_mock()
        gen_smpl_model.generate_with_sampler(top_k=2, top_p=1.0, adjust_fn=adjust_fn, max_tokens=1)
        assert adjust_fn.call_args_list[0][0][0].prev_probs == []

    def test_eval_called_with_prompt_first(self, gen_smpl_model) -> None:
        """The first eval() call in generate_with_sampler receives the formatted prompt tokens."""
        gen_smpl_model.generate_with_sampler(top_k=2, top_p=1.0, adjust_fn=lambda ctx: ctx.token_id_probs, max_tokens=1)
        # context is a mutable list extended by token_ids, so check the leading prompt slice.
        first_call_args = gen_smpl_model._llm.eval.call_args_list[0].args[0]
        assert first_call_args[:3] == [10, 20, 30]

    def test_eval_called_once_per_token_plus_prompt(self, gen_smpl_model) -> None:
        """generate_with_sampler uses incremental eval: once for the prompt then once per token."""
        gen_smpl_model.generate_with_sampler(top_k=2, top_p=1.0, adjust_fn=lambda ctx: ctx.token_id_probs, max_tokens=3)
        # 1 prompt eval + 3 token evals = 4 total
        assert gen_smpl_model._llm.eval.call_count == 4

    def test_single_token_eval_per_step(self, gen_smpl_model) -> None:
        """Each per-token eval() call passes exactly the new token IDs, not the full context."""
        gen_smpl_model.generate_with_sampler(top_k=2, top_p=1.0, adjust_fn=lambda ctx: ctx.token_id_probs, max_tokens=2)
        # Call 0 is the prompt; calls 1+ are single-token evals
        for token_call in gen_smpl_model._llm.eval.call_args_list[1:]:
            assert len(token_call.args[0]) == 1

    def test_max_tokens_zero_skips_generation_steps(self, gen_smpl_model) -> None:
        """With max_tokens=0, no generation step runs and adjust_fn is never called."""
        adjust_fn = MagicMock(return_value=id_logprobs_to_candidate_tokens({1: -0.5}))

        gen_smpl_model.generate_with_sampler(
            top_k=2,
            top_p=1.0,
            adjust_fn=adjust_fn,
            max_tokens=0,
        )

        adjust_fn.assert_not_called()
        gen_smpl_model._llm.eval.assert_called_once()  # prompt eval only

    def test_generation_loop_has_runaway_guard(self, gen_smpl_model) -> None:
        """A guard catches accidental infinite-loop mutants quickly."""
        max_tokens = 3
        gen_smpl_model.generate_with_sampler(
            top_k=1,
            top_p=1.0,
            adjust_fn=lambda ctx: ctx.token_id_probs,
            max_tokens=max_tokens,
        )
        assert gen_smpl_model._llm.eval.call_count == 1 + max_tokens


class TestSampleToken:
    """Tests for Model.sample_token."""

    @pytest.fixture
    def one_step_model(self, model):
        """Model configured for single-step adjusted-token sampling tests."""
        vocab_size = 4
        scores = np.zeros((2048, vocab_size), dtype=np.float32)
        scores[1] = [-2.0, 3.0, 1.0, 0.0]
        scores[2] = [-2.0, 3.0, 1.0, 0.0]
        scores[4] = [-2.0, 3.0, 1.0, 0.0]
        model._llm.scores = scores
        model._llm.n_tokens = 0
        model._logits_all = True
        model._llm.token_eos.return_value = 0
        model._llm.detokenize.side_effect = (
            lambda ids, special=False: f'<tok{ids[0]}>'.encode()
        )
        model._llm.tokenize.return_value = [42]

        def mock_reset():
            model._llm.n_tokens = 0

        def mock_eval(tokens):
            model._llm.n_tokens += len(tokens)

        model._llm.reset.side_effect = mock_reset
        model._llm.eval.side_effect = mock_eval
        return model

    def test_uses_live_state_without_prompt_rebuild(self, one_step_model) -> None:
        """With live tokens present, it does not reset/eval prompt state."""
        one_step_model._llm.n_tokens = 5

        with patch.object(one_step_model, '_format_chat_prompt', side_effect=AssertionError('no prompt rebuild expected')), \
             patch('problm_solver.llama_interface.sample_from_logprobs', return_value='<tok1>'):
            result = one_step_model.sample_token(
                top_k=2,
                top_p=1.0,
                adjust_fn=lambda ctx: ctx.token_id_probs,
                use_live_state=True,
                commit_token=False,
            )

        one_step_model._llm.reset.assert_not_called()
        one_step_model._llm.eval.assert_not_called()
        assert result['state_source'] == 'live'
        assert result['context_tokens_used_for_eval'] is None

    def test_rebuilds_from_prompt_when_live_state_empty(self, one_step_model) -> None:
        """If live state is empty, it falls back to prompt evaluation."""
        one_step_model._llm.n_tokens = 0

        with patch.object(one_step_model, '_format_chat_prompt', return_value=[1, 2]), \
             patch('problm_solver.llama_interface.sample_from_logprobs', return_value='<tok1>'):
            result = one_step_model.sample_token(
                top_k=2,
                top_p=1.0,
                adjust_fn=lambda ctx: ctx.token_id_probs,
                use_live_state=True,
                commit_token=False,
            )

        one_step_model._llm.reset.assert_called_once()
        assert one_step_model._llm.eval.call_args_list[0] == call([1, 2])
        assert result['state_source'] == 'prompt'
        assert result['context_tokens_used_for_eval'] == [1, 2]

    def test_use_live_state_false_rebuilds_even_if_live_exists(self, one_step_model) -> None:
        """use_live_state=False forces rebuild from provided context tokens."""
        one_step_model._llm.n_tokens = 5

        with patch.object(one_step_model, '_format_chat_prompt', side_effect=AssertionError('context tokens should be used')), \
             patch('problm_solver.llama_interface.sample_from_logprobs', return_value='<tok1>'):
            result = one_step_model.sample_token(
                top_k=2,
                top_p=1.0,
                adjust_fn=lambda ctx: ctx.token_id_probs,
                use_live_state=False,
                context_tokens=[7, 8],
                commit_token=False,
            )

        one_step_model._llm.reset.assert_called_once()
        assert one_step_model._llm.eval.call_args_list[0] == call([7, 8])
        assert result['state_source'] == 'context_tokens'
        assert result['context_tokens_used_for_eval'] == [7, 8]

    def test_returns_before_after_candidates_and_sampled_probability(self, one_step_model) -> None:
        """Output contains before/after candidate distributions and sampled token probability."""

        def adjust_fn(ctx):
            ids = ctx.token_id_probs.candidate_ids.copy()
            lps = ctx.token_id_probs.candidate_logprobs.copy()
            lps[1] = lps[1] + 2.0
            return id_logprobs_to_candidate_tokens({int(i): float(lp) for i, lp in zip(ids, lps, strict=True)})

        mock_rng = MagicMock()
        mock_rng.choice.return_value = 1

        with patch.object(one_step_model, '_format_chat_prompt', return_value=[1, 2]), \
             patch('problm_solver.llama_interface.resolve_rng', return_value=mock_rng):
            result = one_step_model.sample_token(
                top_k=2,
                top_p=1.0,
                adjust_fn=adjust_fn,
                use_live_state=False,
                commit_token=False,
            )

        before_tokens = {entry['token'] for entry in result['candidates_before_adjustment']}
        after_tokens = {entry['token'] for entry in result['candidates_after_adjustment']}
        assert before_tokens == {'<tok1>', '<tok2>'}
        assert after_tokens == {'<tok1>', '<tok2>'}

        sampled = result['sampled_token']
        assert sampled is not None
        assert sampled['token'] == '<tok2>'
        assert 0.0 < sampled['prob'] <= 1.0

        sampled_after_prob = next(
            entry['prob']
            for entry in result['candidates_after_adjustment']
            if entry['token'] == '<tok2>'
        )
        assert sampled['prob'] == pytest.approx(sampled_after_prob)

    def test_default_use_live_state_true_does_not_rebuild_prompt(self, one_step_model) -> None:
        """Default call uses live state and does not rebuild prompt when n_tokens>0."""
        one_step_model._llm.n_tokens = 5

        with patch.object(
            one_step_model,
            '_format_chat_prompt',
            side_effect=AssertionError('default should not rebuild prompt'),
        ), patch('problm_solver.llama_interface.sample_from_logprobs', return_value='<tok1>'):
            result = one_step_model.sample_token(
                top_k=2,
                top_p=1.0,
                adjust_fn=lambda ctx: ctx.token_id_probs,
                commit_token=False,
            )

        assert result['state_source'] == 'live'

    def test_default_commit_token_true_commits_non_terminal_token(self, one_step_model) -> None:
        """Default commit_token=True appends non-terminal token via eval(token_ids)."""
        one_step_model._llm.n_tokens = 5
        one_step_model._llm.eval.reset_mock()

        with patch('problm_solver.llama_interface.sample_from_logprobs', return_value='<tok1>'):
            result = one_step_model.sample_token(
                top_k=2,
                top_p=1.0,
                adjust_fn=lambda ctx: ctx.token_id_probs,
            )

        one_step_model._llm.eval.assert_called_once()
        assert len(one_step_model._llm.eval.call_args.args[0]) == 1
        assert result['sampled_token_is_terminal'] is False

    def test_terminal_eos_token_sets_terminal_flag_and_skips_eval(self, one_step_model) -> None:
        """EOS token IDs are terminal, produce sampled_token=None, and are not eval-committed."""
        one_step_model._llm.n_tokens = 5
        one_step_model._llm.token_eos.return_value = 1
        one_step_model._llm.eval.reset_mock()

        result = one_step_model.sample_token(
            top_k=1,
            top_p=1.0,
            adjust_fn=lambda ctx: ctx.token_id_probs,
        )

        one_step_model._llm.eval.assert_not_called()
        assert result['sampled_token'] is None
        assert result['sampled_token_is_terminal'] is True
