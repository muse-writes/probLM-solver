"""Tests for ModelInstance in llama_interface.py."""

from contextlib import ExitStack
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from problm_solver.data import LLMNextTokenData, LLMOutputData, LLMOutputDataFull


def _make_llama_mock(response_text: str = 'Mock response.') -> MagicMock:
    """Return a MagicMock that mimics the low-level llama.cpp API used by ModelInstance."""
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
def model_instance():
    """Return a ModelInstance with a mocked underlying Llama object."""
    from problm_solver.llama_interface import ModelInstance

    with patch('problm_solver.llama_interface.Llama') as MockLlama:
        MockLlama.return_value = _make_llama_mock('Test answer.')
        instance = ModelInstance(fname='fake.gguf', context='What is the answer?')
    return instance


class TestModelInstanceInit:
    """Tests for ModelInstance.__init__."""

    def test_logits_all_defaults_to_false(self) -> None:
        """logits_all defaults to False, so Llama is constructed without it set."""
        from problm_solver.llama_interface import ModelInstance

        with patch('problm_solver.llama_interface.Llama') as MockLlama:
            MockLlama.return_value = _make_llama_mock()
            ModelInstance(fname='fake.gguf', context='Hello')
            _, kwargs = MockLlama.call_args
            assert kwargs.get('logits_all') is False

    def test_logits_all_true_passed_to_llama(self) -> None:
        """logits_all=True is forwarded to the Llama constructor."""
        from problm_solver.llama_interface import ModelInstance

        with patch('problm_solver.llama_interface.Llama') as MockLlama:
            MockLlama.return_value = _make_llama_mock()
            ModelInstance(fname='fake.gguf', context='Hello', logits_all=True)
            _, kwargs = MockLlama.call_args
            assert kwargs.get('logits_all') is True

    def test_cache_sized_from_model_metadata(self) -> None:
        """LlamaRAMCache capacity is derived from model metadata and n_ctx."""
        from problm_solver.llama_interface import ModelInstance

        with patch('problm_solver.llama_interface.Llama') as MockLlama, \
             patch('problm_solver.llama_interface.LlamaRAMCache') as MockCache:
            MockLlama.return_value = _make_llama_mock()
            ModelInstance(fname='fake.gguf', context='Hello')
            _, kwargs = MockCache.call_args
            # 4 states × (2048 ctx × 2 K&V × 32 layers × 8 KV heads × 128 head_dim × 2 bytes)
            assert kwargs.get('capacity_bytes') == 4 * 2048 * 2 * 32 * 8 * 128 * 2


class TestModelInstanceQuery:
    """Tests for ModelInstance.query."""

@pytest.fixture
def low_level_model(model_instance):
    """Extend model_instance with low-level eval state for Phase 5 tests.

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

    model_instance._llm.scores = scores
    model_instance._llm.n_tokens = 0
    model_instance._llm.token_eos.return_value = eos_id

    def mock_detokenize(ids, special=False):
        if special:
            return f'tok{ids[0]}'.encode()
        return b'decoded output'
    model_instance._llm.detokenize.side_effect = mock_detokenize

    def mock_reset():
        model_instance._llm.n_tokens = 0
    model_instance._llm.reset.side_effect = mock_reset

    def mock_eval(tokens):
        model_instance._llm.n_tokens += len(tokens)
    model_instance._llm.eval.side_effect = mock_eval

    with patch.object(model_instance, '_format_chat_prompt', return_value=[1, 2, 3]):
        yield model_instance


class TestModelInstanceQuery:
    """Tests for ModelInstance.query."""

    def test_returns_string(self, low_level_model) -> None:
        """query() returns a plain string."""
        mock_rng = MagicMock()
        mock_rng.gumbel.return_value = np.zeros(4)
        with patch('problm_solver.llama_interface._as_rng', return_value=mock_rng):
            result = low_level_model.query()
        assert isinstance(result, str)

    def test_returns_detokenized_output(self, low_level_model) -> None:
        """query() returns the detokenized form of the generated token IDs."""
        mock_rng = MagicMock()
        mock_rng.gumbel.return_value = np.zeros(4)
        with patch('problm_solver.llama_interface._as_rng', return_value=mock_rng):
            result = low_level_model.query()
        assert result == 'decoded output'

    def test_calls_reset_then_eval_with_prompt(self, low_level_model) -> None:
        """query() calls reset() then eval() with the formatted prompt tokens."""
        mock_rng = MagicMock()
        mock_rng.gumbel.return_value = np.zeros(4)
        with patch('problm_solver.llama_interface._as_rng', return_value=mock_rng):
            low_level_model.query()
        low_level_model._llm.reset.assert_called_once()
        assert low_level_model._llm.eval.call_args_list[0] == call([1, 2, 3])

    def test_stops_at_eos(self, low_level_model) -> None:
        """query() stops and excludes EOS; only the token before EOS is in the output."""
        mock_rng = MagicMock()
        mock_rng.gumbel.return_value = np.zeros(4)
        with patch('problm_solver.llama_interface._as_rng', return_value=mock_rng):
            low_level_model.query()
        # eval: once for prompt, once for the non-EOS token; EOS is not eval'd
        assert low_level_model._llm.eval.call_count == 2


class TestModelInstanceQueryNTimes:
    """Tests for ModelInstance.query_n_times."""

    def test_returns_numpy_array(self, model_instance) -> None:
        """query_n_times() returns a numpy array."""
        with patch.object(model_instance, 'query', return_value='answer'):
            result = model_instance.query_n_times(3)
        assert isinstance(result, np.ndarray)

    def test_array_length_matches_n(self, model_instance) -> None:
        """query_n_times(n) returns exactly n responses."""
        with patch.object(model_instance, 'query', return_value='answer'):
            result = model_instance.query_n_times(5)
        assert len(result) == 5

    def test_query_called_n_times(self, model_instance) -> None:
        """query_n_times(n) calls query() exactly n times."""
        with patch.object(model_instance, 'query', return_value='answer') as mock_q:
            model_instance.query_n_times(4)
        assert mock_q.call_count == 4

    def test_responses_are_strings(self, model_instance) -> None:
        """All elements in the returned array are strings."""
        with patch.object(model_instance, 'query', return_value='answer'):
            result = model_instance.query_n_times(3)
        for item in result:
            assert isinstance(item, str)


class TestModelInstanceQueryLogProbs:
    """Tests for ModelInstance.query_log_probs."""

    def test_returns_llmtokendata(self, low_level_model) -> None:
        """query_log_probs() returns an LLMTokenData instance."""
        from problm_solver.data import LLMTokenData

        mock_rng = MagicMock()
        mock_rng.gumbel.return_value = np.zeros(4)
        with patch('problm_solver.llama_interface._as_rng', return_value=mock_rng):
            result = low_level_model.query_log_probs()
        assert isinstance(result, LLMTokenData)

    def test_tokens_are_strings(self, low_level_model) -> None:
        """Tokens in the returned LLMTokenData are decoded strings."""
        mock_rng = MagicMock()
        mock_rng.gumbel.return_value = np.zeros(4)
        with patch('problm_solver.llama_interface._as_rng', return_value=mock_rng):
            result = low_level_model.query_log_probs()
        assert all(isinstance(t, str) for t in result.tokens)

    def test_probs_are_exp_of_sampled_token_logprobs(self, low_level_model) -> None:
        """Each probability equals exp(log-prob) of the corresponding sampled token."""
        from problm_solver.llama_interface import ModelInstance

        mock_rng = MagicMock()
        mock_rng.gumbel.return_value = np.zeros(4)
        with patch('problm_solver.llama_interface._as_rng', return_value=mock_rng):
            result = low_level_model.query_log_probs()
        lp = ModelInstance._log_softmax(np.array([0.0, 3.0, 1.0, -2.0], dtype=np.float32))
        assert result.probs == pytest.approx([float(np.exp(lp[1]))])

    def test_probs_are_between_zero_and_one(self, low_level_model) -> None:
        """All probabilities are valid (in the range (0, 1])."""
        mock_rng = MagicMock()
        mock_rng.gumbel.return_value = np.zeros(4)
        with patch('problm_solver.llama_interface._as_rng', return_value=mock_rng):
            result = low_level_model.query_log_probs()
        assert all(0.0 < p <= 1.0 for p in result.probs)

    def test_tokens_and_probs_same_length(self, low_level_model) -> None:
        """Tokens and probs are positionally aligned and have equal length."""
        mock_rng = MagicMock()
        mock_rng.gumbel.return_value = np.zeros(4)
        with patch('problm_solver.llama_interface._as_rng', return_value=mock_rng):
            result = low_level_model.query_log_probs()
        assert len(result.tokens) == len(result.probs)

    def test_prompt_is_stored(self, low_level_model) -> None:
        """The prompt is stored on the returned LLMTokenData."""
        mock_rng = MagicMock()
        mock_rng.gumbel.return_value = np.zeros(4)
        with patch('problm_solver.llama_interface._as_rng', return_value=mock_rng):
            result = low_level_model.query_log_probs()
        assert result.prompt == 'What is the answer?'

    def test_stops_at_eos(self, low_level_model) -> None:
        """query_log_probs() accumulates only the tokens generated before EOS."""
        mock_rng = MagicMock()
        mock_rng.gumbel.return_value = np.zeros(4)
        with patch('problm_solver.llama_interface._as_rng', return_value=mock_rng):
            result = low_level_model.query_log_probs()
        # scores[2]=token1 (non-EOS), scores[3]=token3=EOS: exactly 1 token
        assert len(result.tokens) == 1


class TestModelInstanceGenerateData:
    """Tests for ModelInstance.generate_data."""

    def test_returns_llmoutputdata(self, model_instance) -> None:
        """generate_data() returns an LLMOutputData instance."""
        with patch.object(model_instance, 'query', return_value='answer'):
            result = model_instance.generate_data(3)
        assert isinstance(result, LLMOutputData)

    def test_prompt_matches_context(self, model_instance) -> None:
        """generate_data() stores the model's context string as the prompt."""
        with patch.object(model_instance, 'query', return_value='answer'):
            result = model_instance.generate_data(3)
        assert result.prompt == model_instance.context

    def test_data_length_matches_n_samples(self, model_instance) -> None:
        """generate_data(n) produces exactly n responses in the result."""
        with patch.object(model_instance, 'query', return_value='answer'):
            result = model_instance.generate_data(6)
        assert len(result.data) == 6

    def test_written_flag_is_false(self, model_instance) -> None:
        """Freshly generated data has written=False — it hasn't been saved yet."""
        with patch.object(model_instance, 'query', return_value='answer'):
            result = model_instance.generate_data(2)
        assert result.written is False


class TestModelInstanceQueryBranch:
    """Tests for ModelInstance.query_branch."""

    # Vocabulary size and EOS token ID used across the fixture and tests.
    _VOCAB = 5
    _EOS = 4

    @pytest.fixture
    def branch_model(self, model_instance):
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

        model_instance._llm.scores = scores
        model_instance._llm.n_tokens = 0
        model_instance._llm.token_eos.return_value = self._EOS

        n_tokens_snapshot = [0]
        saved_state = MagicMock()

        def mock_reset():
            model_instance._llm.n_tokens = 0
        model_instance._llm.reset.side_effect = mock_reset

        def mock_eval(tokens):
            model_instance._llm.n_tokens += len(tokens)
        model_instance._llm.eval.side_effect = mock_eval

        def mock_save_state():
            n_tokens_snapshot[0] = model_instance._llm.n_tokens
            return saved_state
        model_instance._llm.save_state.side_effect = mock_save_state

        def mock_load_state(state):
            if state is saved_state:
                model_instance._llm.n_tokens = n_tokens_snapshot[0]
        model_instance._llm.load_state.side_effect = mock_load_state

        # Expose the sentinel so tests can assert the exact object passed.
        model_instance._test_saved_state = saved_state
        return model_instance

    def test_returns_float(self, branch_model) -> None:
        """query_branch() returns a Python float."""
        mock_rng = MagicMock()
        mock_rng.gumbel.return_value = np.zeros(self._VOCAB)
        with patch('problm_solver.llama_interface._as_rng', return_value=mock_rng):
            result = branch_model.query_branch([10, 20, 30], max_tokens=1)
        assert isinstance(result, float)

    def test_returns_zero_for_immediate_eos(self, branch_model) -> None:
        """Returns 0.0 when the first sampled token is EOS."""
        # Make EOS the argmax by giving it an overwhelming logit.
        branch_model._llm.scores[2] = [-10.0, -10.0, -10.0, -10.0, 10.0]
        mock_rng = MagicMock()
        mock_rng.gumbel.return_value = np.zeros(self._VOCAB)
        with patch('problm_solver.llama_interface._as_rng', return_value=mock_rng):
            result = branch_model.query_branch([10, 20, 30], max_tokens=5)
        assert result == 0.0

    def test_sums_log_probs_of_generated_tokens(self, branch_model) -> None:
        """Return value equals the sum of log-probs of the sampled tokens."""
        from problm_solver.llama_interface import ModelInstance

        # With Gumbel noise = 0, sampling is greedy: argmax of logprobs.
        # Step 1: scores[2], argmax = 1 (logit 3.0)
        # Step 2: scores[3], argmax = 2 (logit 2.0)
        lp1 = float(ModelInstance._log_softmax(
            np.array([0.5, 3.0, 1.5, 0.2, -2.0], dtype=np.float32)
        )[1])
        lp2 = float(ModelInstance._log_softmax(
            np.array([0.5, 0.5, 2.0, 0.2, -2.0], dtype=np.float32)
        )[2])
        mock_rng = MagicMock()
        mock_rng.gumbel.return_value = np.zeros(self._VOCAB)
        with patch('problm_solver.llama_interface._as_rng', return_value=mock_rng):
            result = branch_model.query_branch([10, 20, 30], max_tokens=2)
        assert result == pytest.approx(lp1 + lp2)

    def test_stops_at_max_tokens(self, branch_model) -> None:
        """Generation stops after exactly max_tokens tokens when EOS never appears."""
        mock_rng = MagicMock()
        mock_rng.gumbel.return_value = np.zeros(self._VOCAB)
        with patch('problm_solver.llama_interface._as_rng', return_value=mock_rng):
            branch_model.query_branch([10, 20, 30], max_tokens=2)
        # eval: once for context, once per generated token (2 tokens)
        assert branch_model._llm.eval.call_count == 3

    def test_eos_log_prob_not_included_in_sum(self, branch_model) -> None:
        """The log-probability of the EOS token itself is not added to the total."""
        from problm_solver.llama_interface import ModelInstance

        # Step 1 generates token 1 (non-EOS); step 2 generates EOS.
        branch_model._llm.scores[3] = [-10.0, -10.0, -10.0, -10.0, 10.0]  # EOS argmax
        lp1 = float(ModelInstance._log_softmax(
            np.array([0.5, 3.0, 1.5, 0.2, -2.0], dtype=np.float32)
        )[1])
        mock_rng = MagicMock()
        mock_rng.gumbel.return_value = np.zeros(self._VOCAB)
        with patch('problm_solver.llama_interface._as_rng', return_value=mock_rng):
            result = branch_model.query_branch([10, 20, 30], max_tokens=5)
        assert result == pytest.approx(lp1)

    def test_calls_reset(self, branch_model) -> None:
        """reset() is called once to clear stale KV-cache state."""
        mock_rng = MagicMock()
        mock_rng.gumbel.return_value = np.zeros(self._VOCAB)
        with patch('problm_solver.llama_interface._as_rng', return_value=mock_rng):
            branch_model.query_branch([10, 20, 30], max_tokens=1)
        branch_model._llm.reset.assert_called_once()

    def test_calls_eval_with_context_tokens(self, branch_model) -> None:
        """eval() is first called with the full context token list."""
        context = [10, 20, 30]
        mock_rng = MagicMock()
        mock_rng.gumbel.return_value = np.zeros(self._VOCAB)
        with patch('problm_solver.llama_interface._as_rng', return_value=mock_rng):
            branch_model.query_branch(context, max_tokens=1)
        assert branch_model._llm.eval.call_args_list[0] == call(context)

    def test_saves_state_after_context_eval(self, branch_model) -> None:
        """save_state() is called exactly once, after evaluating the context."""
        mock_rng = MagicMock()
        mock_rng.gumbel.return_value = np.zeros(self._VOCAB)
        with patch('problm_solver.llama_interface._as_rng', return_value=mock_rng):
            branch_model.query_branch([10, 20, 30], max_tokens=1)
        branch_model._llm.save_state.assert_called_once()
        # The n_tokens captured at save time equals len(context_tokens).
        assert branch_model._llm.n_tokens >= 3  # at least context + 1 generated

    def test_loads_saved_state(self, branch_model) -> None:
        """load_state() is called with exactly the object returned by save_state()."""
        mock_rng = MagicMock()
        mock_rng.gumbel.return_value = np.zeros(self._VOCAB)
        with patch('problm_solver.llama_interface._as_rng', return_value=mock_rng):
            branch_model.query_branch([10, 20, 30], max_tokens=1)
        branch_model._llm.load_state.assert_called_once_with(
            branch_model._test_saved_state
        )

    def test_eval_called_once_per_generated_token(self, branch_model) -> None:
        """eval() is called once for the context and once for each generated token."""
        mock_rng = MagicMock()
        mock_rng.gumbel.return_value = np.zeros(self._VOCAB)
        with patch('problm_solver.llama_interface._as_rng', return_value=mock_rng):
            branch_model.query_branch([10, 20, 30], max_tokens=3)
        # 1 context eval + 3 single-token evals
        assert branch_model._llm.eval.call_count == 4


class TestQueryLogProbsNextToken:
    """Tests for ModelInstance.query_log_probs_next_token."""

    @pytest.fixture
    def next_token_model(self, model_instance):
        """Configure mock LLM for query_log_probs_next_token.

        scores[n_tokens - 1] is the logit row used, so after reset() +
        eval([1, 2, 3]) n_tokens=3 and the test values sit at scores[2].
        """
        vocab_size = 5
        scores = np.zeros((2048, vocab_size), dtype=np.float32)
        scores[2] = [0.0, 3.0, 1.0, 2.0, 0.5]  # token 1 highest, token 3 second
        model_instance._llm.scores = scores
        model_instance._llm.n_tokens = 0
        model_instance._llm.detokenize.side_effect = (
            lambda ids, special=False: f'<tok{ids[0]}>'.encode()
        )

        def mock_reset():
            model_instance._llm.n_tokens = 0
        model_instance._llm.reset.side_effect = mock_reset

        def mock_eval(tokens):
            model_instance._llm.n_tokens += len(tokens)
        model_instance._llm.eval.side_effect = mock_eval

        return model_instance

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
    """Tests for ModelInstance._format_chat_prompt."""

    @pytest.fixture
    def chat_prompt_model(self, model_instance):
        """Configure the mock LLM for _format_chat_prompt."""
        mock_result = MagicMock()
        mock_result.prompt = 'formatted prompt string'
        model_instance._llm.metadata = {'tokenizer.chat_template': 'dummy_template'}
        model_instance._llm.token_eos.return_value = 2
        model_instance._llm.token_bos.return_value = 1
        model_instance._llm.detokenize.return_value = b''
        model_instance._llm.tokenize.return_value = [1, 2, 3, 4, 5]
        with patch(
            'problm_solver.llama_interface.Jinja2ChatFormatter'
        ) as mock_jinja:
            mock_jinja.return_value.return_value = mock_result
            yield model_instance

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
    """Tests for ModelInstance._log_softmax."""

    def test_output_is_valid_log_probability_distribution(self) -> None:
        """exp(log_softmax(x)) sums to 1.0 over the full vocabulary."""
        from problm_solver.llama_interface import ModelInstance

        logits = np.array([1.0, 2.0, 0.5, -1.0], dtype=np.float32)
        result = ModelInstance._log_softmax(logits)
        assert np.exp(result).sum() == pytest.approx(1.0)

    def test_argmax_is_preserved(self) -> None:
        """The token with the highest logit has the highest log-probability."""
        from problm_solver.llama_interface import ModelInstance

        logits = np.array([0.1, 3.0, -1.0, 0.5], dtype=np.float32)
        result = ModelInstance._log_softmax(logits)
        assert np.argmax(result) == np.argmax(logits)

    def test_all_values_are_non_positive(self) -> None:
        """All log-probabilities are ≤ 0 (probabilities are in (0, 1])."""
        from problm_solver.llama_interface import ModelInstance

        logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = ModelInstance._log_softmax(logits)
        assert np.all(result <= 0.0)

    def test_returns_float64_array(self) -> None:
        """Output dtype is float64 regardless of the input dtype."""
        from problm_solver.llama_interface import ModelInstance

        logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = ModelInstance._log_softmax(logits)
        assert result.dtype == np.float64

    def test_output_shape_matches_input(self) -> None:
        """Output array has the same shape as the input logits."""
        from problm_solver.llama_interface import ModelInstance

        logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        result = ModelInstance._log_softmax(logits)
        assert result.shape == logits.shape

    def test_numerically_stable_with_large_logits(self) -> None:
        """Does not overflow when logits are in the hundreds, as is common for LLMs."""
        from problm_solver.llama_interface import ModelInstance

        logits = np.array([300.0, 200.0, 100.0], dtype=np.float32)
        result = ModelInstance._log_softmax(logits)
        assert np.all(np.isfinite(result))
        assert np.exp(result).sum() == pytest.approx(1.0)

    def test_uniform_logits_produce_equal_log_probs(self) -> None:
        """All-equal logits map to the same log-probability for every token."""
        from problm_solver.llama_interface import ModelInstance

        logits = np.full(5, 2.0, dtype=np.float32)
        result = ModelInstance._log_softmax(logits)
        assert np.allclose(result, result[0])


class TestTopKFromLogprobs:
    """Tests for ModelInstance._top_k_from_logprobs."""

    @pytest.fixture
    def logprob_model(self, model_instance):
        """Configure detokenize to return '<tokN>' for token ID N."""
        model_instance._llm.detokenize.side_effect = (
            lambda ids, special=False: f'<tok{ids[0]}>'.encode()
        )
        return model_instance

    def test_returns_exactly_n_entries(self, logprob_model) -> None:
        """The returned dict has exactly n entries."""
        logprobs = np.array([-3.0, -1.0, -0.5, -2.0, -4.0], dtype=np.float64)
        result = logprob_model._top_k_from_logprobs(logprobs, n=3)
        assert len(result) == 3

    def test_contains_highest_logprob_tokens(self, logprob_model) -> None:
        """Result contains the n tokens with the highest log-probabilities."""
        logprobs = np.array([-3.0, -1.0, -0.5, -2.0, -4.0], dtype=np.float64)
        result = logprob_model._top_k_from_logprobs(logprobs, n=2)
        # Top 2: index 2 (-0.5) and index 1 (-1.0)
        assert '<tok2>' in result
        assert '<tok1>' in result

    def test_excludes_lower_logprob_tokens(self, logprob_model) -> None:
        """Tokens outside the top-n are not present in the result."""
        logprobs = np.array([-3.0, -1.0, -0.5, -2.0, -4.0], dtype=np.float64)
        result = logprob_model._top_k_from_logprobs(logprobs, n=2)
        assert '<tok0>' not in result
        assert '<tok3>' not in result
        assert '<tok4>' not in result

    def test_values_match_logprobs_of_their_tokens(self, logprob_model) -> None:
        """Each dict value equals the log-probability at the corresponding vocab index."""
        logprobs = np.array([-3.0, -1.0, -0.5, -2.0, -4.0], dtype=np.float64)
        result = logprob_model._top_k_from_logprobs(logprobs, n=3)
        assert result['<tok2>'] == pytest.approx(-0.5)
        assert result['<tok1>'] == pytest.approx(-1.0)
        assert result['<tok3>'] == pytest.approx(-2.0)

    def test_values_are_python_floats(self, logprob_model) -> None:
        """All values are plain Python floats, not numpy scalars."""
        logprobs = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
        result = logprob_model._top_k_from_logprobs(logprobs, n=2)
        assert all(type(v) is float for v in result.values())

    def test_sorted_descending_by_log_prob(self, logprob_model) -> None:
        """Keys are ordered from highest to lowest log-probability (dict insertion order)."""
        logprobs = np.array([-3.0, -1.0, -0.5, -2.0, -4.0], dtype=np.float64)
        result = logprob_model._top_k_from_logprobs(logprobs, n=3)
        values = list(result.values())
        assert values == sorted(values, reverse=True)

    def test_n_clamped_to_vocab_size(self, logprob_model) -> None:
        """Requesting more tokens than vocab size returns every token."""
        logprobs = np.array([-1.0, -2.0, -3.0], dtype=np.float64)
        result = logprob_model._top_k_from_logprobs(logprobs, n=100)
        assert len(result) == 3

    def test_n_of_one_returns_single_highest_token(self, logprob_model) -> None:
        """n=1 returns only the argmax token with its log-probability."""
        logprobs = np.array([-3.0, -0.1, -2.0], dtype=np.float64)
        result = logprob_model._top_k_from_logprobs(logprobs, n=1)
        assert list(result.keys()) == ['<tok1>']
        assert result['<tok1>'] == pytest.approx(-0.1)


@pytest.fixture
def gen_adj_model(model_instance):
    """ModelInstance with all generate_adjusted() dependencies mocked.

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

    model_instance._llm.scores = scores
    model_instance._llm.n_tokens = 0
    model_instance._logits_all = True
    model_instance._llm.token_eos.return_value = 0
    model_instance._llm.tokenize.return_value = [42]
    model_instance._llm.detokenize.side_effect = (
        lambda ids, special=False: f'<tok{ids[0]}>'.encode()
    )

    def mock_reset():
        model_instance._llm.n_tokens = 0
    model_instance._llm.reset.side_effect = mock_reset

    def mock_eval(tokens):
        model_instance._llm.n_tokens += len(tokens)
    model_instance._llm.eval.side_effect = mock_eval

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(model_instance, '_format_chat_prompt', return_value=[10, 20, 30])
        )
        stack.enter_context(
            patch('problm_solver.llama_interface.sample_from_logprobs', return_value=' hello')
        )
        stack.enter_context(
            patch('problm_solver.llama_interface.prob_of_token', return_value=0.8)
        )
        yield model_instance


class TestGenerateAdjusted:
    """Tests for ModelInstance.generate_adjusted."""

    def test_returns_llmoutputdatafull(self, gen_adj_model) -> None:
        """generate_adjusted() returns an LLMOutputDataFull instance."""
        result = gen_adj_model.generate_adjusted(top_k=2, top_p=1.0, adjust_fn=lambda ctx: ctx.token_probs, max_tokens=3)
        assert isinstance(result, LLMOutputDataFull)

    def test_context_is_list_of_strings(self, gen_adj_model) -> None:
        """Context on the returned LLMOutputDataFull is a list of strings."""
        result = gen_adj_model.generate_adjusted(top_k=2, top_p=1.0, adjust_fn=lambda ctx: ctx.token_probs, max_tokens=3)
        assert isinstance(result.context, list)
        assert all(isinstance(s, str) for s in result.context)

    def test_written_flag_is_false(self, gen_adj_model) -> None:
        """Freshly generated data has _written=False."""
        result = gen_adj_model.generate_adjusted(top_k=2, top_p=1.0, adjust_fn=lambda ctx: ctx.token_probs, max_tokens=3)
        assert result._written is False

    def test_loops_exactly_max_tokens_times(self, gen_adj_model) -> None:
        """eval() is called once for the prompt then once per generated token."""
        gen_adj_model.generate_adjusted(top_k=2, top_p=1.0, adjust_fn=lambda ctx: ctx.token_probs, max_tokens=4)
        # 1 prompt eval + 4 token evals
        assert gen_adj_model._llm.eval.call_count == 5

    def test_adjust_fn_called_each_step(self, gen_adj_model) -> None:
        """adjust_fn is called once per generated token."""
        adjust_fn = MagicMock(return_value={' hello': -0.5})
        gen_adj_model.generate_adjusted(top_k=2, top_p=1.0, adjust_fn=adjust_fn, max_tokens=3)
        assert adjust_fn.call_count == 3

    def test_adjust_fn_receives_top_k_tokens(self, gen_adj_model) -> None:
        """adjust_fn receives a GenerationContext whose token_probs is built from scores."""
        from problm_solver.llama_interface import ModelInstance

        adjust_fn = MagicMock(return_value={' hello': -0.5})
        gen_adj_model.generate_adjusted(top_k=2, top_p=1.0, adjust_fn=adjust_fn, max_tokens=1)
        ctx = adjust_fn.call_args[0][0]
        # scores[2] = [-10, 3, 1, 0.5]; top-2 are '<tok1>' and '<tok2>'
        lp = ModelInstance._log_softmax(np.array([-10.0, 3.0, 1.0, 0.5], dtype=np.float32))
        assert ctx.token_probs == pytest.approx({'<tok1>': float(lp[1]), '<tok2>': float(lp[2])})

    def test_adjust_fn_receives_empty_prev_probs_on_first_step(self, gen_adj_model) -> None:
        """adjust_fn receives a GenerationContext with empty prev_probs on the first step."""
        adjust_fn = MagicMock(return_value={' hello': -0.5})
        gen_adj_model.generate_adjusted(top_k=2, top_p=1.0, adjust_fn=adjust_fn, max_tokens=1)
        ctx = adjust_fn.call_args_list[0][0][0]
        assert ctx.prev_probs == []

    def test_adjust_fn_receives_growing_prev_probs(self, gen_adj_model) -> None:
        """prev_probs grows by one entry per step, containing prob_of_token return values."""
        adjust_fn = MagicMock(return_value={' hello': -0.5})
        gen_adj_model.generate_adjusted(top_k=2, top_p=1.0, adjust_fn=adjust_fn, max_tokens=3)
        # Step 1: prev_probs = []; step 2: [0.8]; step 3: [0.8, 0.8]
        assert adjust_fn.call_args_list[0][0][0].prev_probs == []
        assert adjust_fn.call_args_list[1][0][0].prev_probs == [0.8]
        assert adjust_fn.call_args_list[2][0][0].prev_probs == [0.8, 0.8]

    def test_response_topk_tokens_are_sampled_tokens(self, gen_adj_model) -> None:
        """response_topk[0] contains the token strings chosen at each step."""
        result = gen_adj_model.generate_adjusted(top_k=2, top_p=1.0, adjust_fn=lambda ctx: ctx.token_probs, max_tokens=2)
        assert result.response_topk[0] == [' hello', ' hello']

    def test_stops_early_on_eos_token(self, gen_adj_model) -> None:
        """The loop breaks before max_tokens when tokenize returns the EOS token ID."""
        gen_adj_model._llm.tokenize.return_value = [0]
        gen_adj_model.generate_adjusted(top_k=2, top_p=1.0, adjust_fn=lambda ctx: ctx.token_probs, max_tokens=10)
        # Only the prompt eval ran; no token eval because first sample was EOS
        assert gen_adj_model._llm.eval.call_count == 1

    def test_stops_early_on_empty_token_ids(self, gen_adj_model) -> None:
        """The loop breaks when tokenize returns an empty list for the sampled token."""
        gen_adj_model._llm.tokenize.return_value = []
        gen_adj_model.generate_adjusted(top_k=2, top_p=1.0, adjust_fn=lambda ctx: ctx.token_probs, max_tokens=10)
        assert gen_adj_model._llm.eval.call_count == 1

    def test_prev_probs_reset_between_calls(self, gen_adj_model) -> None:
        """prev_probs starts empty on every call to generate_adjusted, not carried over."""
        adjust_fn = MagicMock(return_value={' hello': -0.5})
        gen_adj_model.generate_adjusted(top_k=2, top_p=1.0, adjust_fn=adjust_fn, max_tokens=2)
        gen_adj_model._llm.eval.reset_mock()
        adjust_fn.reset_mock()
        gen_adj_model.generate_adjusted(top_k=2, top_p=1.0, adjust_fn=adjust_fn, max_tokens=1)
        assert adjust_fn.call_args_list[0][0][0].prev_probs == []

    def test_eval_called_with_prompt_first(self, gen_adj_model) -> None:
        """The first eval() call in generate_adjusted receives the formatted prompt tokens."""
        gen_adj_model.generate_adjusted(top_k=2, top_p=1.0, adjust_fn=lambda ctx: ctx.token_probs, max_tokens=1)
        # context is a mutable list extended by token_ids, so check the leading prompt slice.
        first_call_args = gen_adj_model._llm.eval.call_args_list[0].args[0]
        assert first_call_args[:3] == [10, 20, 30]

    def test_eval_called_once_per_token_plus_prompt(self, gen_adj_model) -> None:
        """generate_adjusted uses incremental eval: once for the prompt then once per token."""
        gen_adj_model.generate_adjusted(top_k=2, top_p=1.0, adjust_fn=lambda ctx: ctx.token_probs, max_tokens=3)
        # 1 prompt eval + 3 token evals = 4 total
        assert gen_adj_model._llm.eval.call_count == 4

    def test_single_token_eval_per_step(self, gen_adj_model) -> None:
        """Each per-token eval() call passes exactly the new token IDs, not the full context."""
        gen_adj_model.generate_adjusted(top_k=2, top_p=1.0, adjust_fn=lambda ctx: ctx.token_probs, max_tokens=2)
        # Call 0 is the prompt; calls 1+ are single-token evals
        for token_call in gen_adj_model._llm.eval.call_args_list[1:]:
            assert token_call == call([42])  # tokenize() returns [42]

    def test_max_tokens_zero_skips_generation_steps(self, gen_adj_model) -> None:
        """With max_tokens=0, no generation step runs and adjust_fn is never called."""
        adjust_fn = MagicMock(return_value={' hello': -0.5})

        gen_adj_model.generate_adjusted(
            top_k=2,
            top_p=1.0,
            adjust_fn=adjust_fn,
            max_tokens=0,
        )

        adjust_fn.assert_not_called()
        gen_adj_model._llm.eval.assert_called_once()  # prompt eval only

    def test_generation_loop_has_runaway_guard(self, gen_adj_model) -> None:
        """A guard catches accidental infinite-loop mutants quickly."""
        max_tokens = 3
        sample_calls = 0

        def guarded_sample(_: dict[str, float]) -> str:
            nonlocal sample_calls
            sample_calls += 1
            if sample_calls > max_tokens:
                msg = 'runaway loop: sampled more than max_tokens'
                raise AssertionError(msg)
            return ' hello'

        with patch('problm_solver.llama_interface.sample_from_logprobs', side_effect=guarded_sample):
            gen_adj_model.generate_adjusted(
                top_k=2,
                top_p=1.0,
                adjust_fn=lambda ctx: ctx.token_probs,
                max_tokens=max_tokens,
            )

        assert sample_calls == max_tokens


class TestSampleTokenAdjusted:
    """Tests for ModelInstance.sample_token_adjusted."""

    @pytest.fixture
    def one_step_model(self, model_instance):
        """Model configured for single-step adjusted-token sampling tests."""
        vocab_size = 4
        scores = np.zeros((2048, vocab_size), dtype=np.float32)
        scores[1] = [-2.0, 3.0, 1.0, 0.0]
        scores[2] = [-2.0, 3.0, 1.0, 0.0]
        scores[4] = [-2.0, 3.0, 1.0, 0.0]
        model_instance._llm.scores = scores
        model_instance._llm.n_tokens = 0
        model_instance._logits_all = True
        model_instance._llm.token_eos.return_value = 0
        model_instance._llm.detokenize.side_effect = (
            lambda ids, special=False: f'<tok{ids[0]}>'.encode()
        )
        model_instance._llm.tokenize.return_value = [42]

        def mock_reset():
            model_instance._llm.n_tokens = 0

        def mock_eval(tokens):
            model_instance._llm.n_tokens += len(tokens)

        model_instance._llm.reset.side_effect = mock_reset
        model_instance._llm.eval.side_effect = mock_eval
        return model_instance

    def test_uses_live_state_without_prompt_rebuild(self, one_step_model) -> None:
        """With live tokens present, it does not reset/eval prompt state."""
        one_step_model._llm.n_tokens = 5

        with patch.object(one_step_model, '_format_chat_prompt', side_effect=AssertionError('no prompt rebuild expected')), \
             patch('problm_solver.llama_interface.sample_from_logprobs', return_value='<tok1>'):
            result = one_step_model.sample_token_adjusted(
                top_k=2,
                top_p=1.0,
                adjust_fn=lambda ctx: ctx.token_probs,
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
            result = one_step_model.sample_token_adjusted(
                top_k=2,
                top_p=1.0,
                adjust_fn=lambda ctx: ctx.token_probs,
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
            result = one_step_model.sample_token_adjusted(
                top_k=2,
                top_p=1.0,
                adjust_fn=lambda ctx: ctx.token_probs,
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
            adjusted = dict(ctx.token_probs)
            adjusted['<tok2>'] = adjusted['<tok2>'] + 2.0
            return adjusted

        with patch.object(one_step_model, '_format_chat_prompt', return_value=[1, 2]), \
             patch('problm_solver.llama_interface.sample_from_logprobs', return_value='<tok2>'):
            result = one_step_model.sample_token_adjusted(
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
            result = one_step_model.sample_token_adjusted(
                top_k=2,
                top_p=1.0,
                adjust_fn=lambda ctx: ctx.token_probs,
                commit_token=False,
            )

        assert result['state_source'] == 'live'

    def test_default_commit_token_true_commits_non_terminal_token(self, one_step_model) -> None:
        """Default commit_token=True appends non-terminal token via eval(token_ids)."""
        one_step_model._llm.n_tokens = 5
        one_step_model._llm.eval.reset_mock()

        with patch('problm_solver.llama_interface.sample_from_logprobs', return_value='<tok1>'):
            result = one_step_model.sample_token_adjusted(
                top_k=2,
                top_p=1.0,
                adjust_fn=lambda ctx: ctx.token_probs,
            )

        one_step_model._llm.eval.assert_called_once_with([42])
        assert result['sampled_token_is_terminal'] is False

    def test_terminal_eos_token_sets_terminal_flag_and_skips_eval(self, one_step_model) -> None:
        """EOS token IDs are terminal, produce sampled_token=None, and are not eval-committed."""
        one_step_model._llm.n_tokens = 5
        one_step_model._llm.tokenize.return_value = [0]  # EOS
        one_step_model._llm.eval.reset_mock()

        with patch('problm_solver.llama_interface.sample_from_logprobs', return_value='<tok1>'):
            result = one_step_model.sample_token_adjusted(
                top_k=2,
                top_p=1.0,
                adjust_fn=lambda ctx: ctx.token_probs,
            )

        one_step_model._llm.eval.assert_not_called()
        assert result['sampled_token'] is None
        assert result['sampled_token_is_terminal'] is True
