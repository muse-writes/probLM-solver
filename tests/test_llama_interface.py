"""Tests for ModelInstance in llama_interface.py."""

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from problm_solver.data import LLMNextTokenData, LLMOutputData


def _make_llama_mock(response_text: str = 'Mock response.') -> MagicMock:
    """Return a MagicMock that mimics the Llama chat completion API."""
    mock_llm = MagicMock()
    mock_llm.create_chat_completion.return_value = {
        'choices': [{'message': {'content': response_text}}]
    }
    mock_llm.metadata = {
        'general.architecture': 'llama',
        'llama.block_count': '32',
        'llama.attention.head_count_kv': '8',
        'llama.attention.head_count': '32',
        'llama.embedding_length': '4096',
    }
    mock_llm.n_ctx.return_value = 2048
    return mock_llm


@pytest.fixture()
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

    def test_query_returns_string(self, model_instance) -> None:
        """query() returns a plain string."""
        result = model_instance.query()
        assert isinstance(result, str)

    def test_query_returns_llm_content(self, model_instance) -> None:
        """query() returns the content field from the chat completion response."""
        result = model_instance.query()
        assert result == 'Test answer.'

    def test_query_calls_create_chat_completion(self, model_instance) -> None:
        """query() delegates to create_chat_completion on the underlying Llama object."""
        model_instance.query()
        model_instance._llm.create_chat_completion.assert_called_once()

    def test_query_sends_context_as_user_message(self, model_instance) -> None:
        """query() sends the stored context as a user-role message."""
        model_instance.query()
        call_kwargs = model_instance._llm.create_chat_completion.call_args
        messages = call_kwargs.kwargs.get('messages') or call_kwargs.args[0]
        assert messages[0]['role'] == 'user'
        assert messages[0]['content'] == 'What is the answer?'


class TestModelInstanceQueryNTimes:
    """Tests for ModelInstance.query_n_times."""

    def test_returns_numpy_array(self, model_instance) -> None:
        """query_n_times() returns a numpy array."""
        result = model_instance.query_n_times(3)
        assert isinstance(result, np.ndarray)

    def test_array_length_matches_n(self, model_instance) -> None:
        """query_n_times(n) returns exactly n responses."""
        result = model_instance.query_n_times(5)
        assert len(result) == 5

    def test_query_called_n_times(self, model_instance) -> None:
        """query_n_times(n) calls the LLM backend exactly n times."""
        model_instance.query_n_times(4)
        assert model_instance._llm.create_chat_completion.call_count == 4

    def test_responses_are_strings(self, model_instance) -> None:
        """All elements in the returned array are strings."""
        result = model_instance.query_n_times(3)
        for item in result:
            assert isinstance(item, str)


class TestModelInstanceQueryLogProbs:
    """Tests for ModelInstance.query_log_probs."""

    @pytest.fixture()
    def logprob_model_instance(self):
        """Return a ModelInstance whose Llama mock returns a logprobs response."""
        from problm_solver.llama_interface import ModelInstance

        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = {
            'choices': [{
                'message': {'content': 'Four.'},
                'logprobs': {
                    'content': [
                        {'token': 'Four', 'logprob': -0.105, 'bytes': None, 'top_logprobs': []},
                        {'token': '.', 'logprob': -0.011, 'bytes': None, 'top_logprobs': []},
                    ]
                },
                'finish_reason': 'stop',
            }]
        }
        with patch('problm_solver.llama_interface.Llama') as MockLlama:
            MockLlama.return_value = mock_llm
            instance = ModelInstance(fname='fake.gguf', context='What is 2+2?')
        return instance

    def test_returns_llmtokendata(self, logprob_model_instance) -> None:
        """query_log_probs() returns an LLMTokenData instance."""
        from problm_solver.data import LLMTokenData

        result = logprob_model_instance.query_log_probs()
        assert isinstance(result, LLMTokenData)

    def test_tokens_extracted_from_logprobs_content(self, logprob_model_instance) -> None:
        """Tokens come from the logprobs content, not from re-tokenizing the text."""
        result = logprob_model_instance.query_log_probs()
        assert result.tokens == ['Four', '.']

    def test_probs_are_exp_of_logprobs(self, logprob_model_instance) -> None:
        """Each probability is exp(logprob) of the corresponding entry."""
        import math

        result = logprob_model_instance.query_log_probs()
        expected = [math.exp(-0.105), math.exp(-0.011)]
        assert result.probs == pytest.approx(expected)

    def test_probs_are_between_zero_and_one(self, logprob_model_instance) -> None:
        """All probabilities are valid (in the range (0, 1])."""
        result = logprob_model_instance.query_log_probs()
        assert all(0.0 < p <= 1.0 for p in result.probs)

    def test_tokens_and_probs_same_length(self, logprob_model_instance) -> None:
        """tokens and probs are positionally aligned and have equal length."""
        result = logprob_model_instance.query_log_probs()
        assert len(result.tokens) == len(result.probs)

    def test_prompt_is_stored(self, logprob_model_instance) -> None:
        """The prompt is stored on the returned LLMTokenData."""
        result = logprob_model_instance.query_log_probs()
        assert result.prompt == 'What is 2+2?'

    def test_passes_top_logprobs_1_to_api(self, logprob_model_instance) -> None:
        """query_log_probs() passes top_logprobs=1 so the chat handler enables logprob output."""
        logprob_model_instance.query_log_probs()
        _, kwargs = logprob_model_instance._llm.create_chat_completion.call_args
        assert kwargs.get('top_logprobs') == 1


class TestModelInstanceGetTokenizer:
    """Tests for ModelInstance.get_tokenizer."""

    def test_returns_llama_tokenizer(self, model_instance) -> None:
        """get_tokenizer() returns a LlamaTokenizer instance."""
        from problm_solver.analysis.tokenizer import LlamaTokenizer

        result = model_instance.get_tokenizer()
        assert isinstance(result, LlamaTokenizer)

    def test_tokenizer_is_backed_by_model_llm(self, model_instance) -> None:
        """The returned tokenizer wraps the same Llama object as the model."""
        from problm_solver.analysis.tokenizer import LlamaTokenizer

        result = model_instance.get_tokenizer()
        assert result._llama is model_instance._llm


class TestModelInstanceGenerateData:
    """Tests for ModelInstance.generate_data."""

    def test_returns_llmoutputdata(self, model_instance) -> None:
        """generate_data() returns an LLMOutputData instance."""
        result = model_instance.generate_data(3)
        assert isinstance(result, LLMOutputData)

    def test_prompt_matches_context(self, model_instance) -> None:
        """generate_data() stores the model's context string as the prompt."""
        result = model_instance.generate_data(3)
        assert result.prompt == model_instance.context

    def test_data_length_matches_n_samples(self, model_instance) -> None:
        """generate_data(n) produces exactly n responses in the result."""
        result = model_instance.generate_data(6)
        assert len(result.data) == 6

    def test_written_flag_is_false(self, model_instance) -> None:
        """Freshly generated data has written=False — it hasn't been saved yet."""
        result = model_instance.generate_data(2)
        assert result.written is False


def _make_next_token_completion(top_logprobs: dict) -> dict:
    """Build a create_completion return value with the given top_logprobs dict."""
    return {
        'choices': [{
            'logprobs': {'top_logprobs': [top_logprobs]},
            'finish_reason': 'length',
        }]
    }


def _make_branch_completion(token_logprobs: list) -> dict:
    """Build a create_completion return value with the given token_logprobs list."""
    return {
        'choices': [{
            'logprobs': {
                'token_logprobs': token_logprobs,
                'tokens': ['token'] * len(token_logprobs),
            },
            'finish_reason': 'length' if token_logprobs else 'stop',
        }]
    }


class TestModelInstanceQueryBranch:
    """Tests for ModelInstance.query_branch."""

    @pytest.fixture()
    def branch_model(self, model_instance):
        """Configure the mock LLM to return a branch completion response."""
        model_instance._llm.create_completion.return_value = _make_branch_completion(
            [-0.5, -1.0, -0.3]
        )
        return model_instance

    def test_returns_float(self, branch_model) -> None:
        """query_branch() returns a float."""
        result = branch_model.query_branch([1, 2, 3], max_tokens=3)
        assert isinstance(result, float)

    def test_sums_token_logprobs(self, branch_model) -> None:
        """result equals the sum of all token_logprobs entries."""
        result = branch_model.query_branch([1, 2, 3], max_tokens=3)
        assert result == pytest.approx(-0.5 + -1.0 + -0.3)

    def test_returns_zero_for_immediate_eos(self, model_instance) -> None:
        """Returns 0.0 when token_logprobs is empty (model generates EOS immediately)."""
        model_instance._llm.create_completion.return_value = _make_branch_completion([])
        result = model_instance.query_branch([1, 2, 3], max_tokens=5)
        assert result == 0.0

    def test_partial_branch_sums_available_logprobs(self, model_instance) -> None:
        """When EOS stops generation early, only the available logprobs are summed."""
        model_instance._llm.create_completion.return_value = _make_branch_completion(
            [-0.5, -1.0]  # only 2 of the requested 5 tokens were generated
        )
        result = model_instance.query_branch([1, 2, 3], max_tokens=5)
        assert result == pytest.approx(-0.5 + -1.0)

    def test_passes_context_tokens_as_prompt(self, branch_model) -> None:
        """create_completion receives context_tokens as its positional argument."""
        context = [10, 20, 30]
        branch_model.query_branch(context, max_tokens=3)
        args, _ = branch_model._llm.create_completion.call_args
        assert args[0] == context

    def test_passes_max_tokens(self, branch_model) -> None:
        """create_completion is called with the correct max_tokens."""
        branch_model.query_branch([1, 2, 3], max_tokens=7)
        _, kwargs = branch_model._llm.create_completion.call_args
        assert kwargs.get('max_tokens') == 7

    def test_passes_logprobs_1(self, branch_model) -> None:
        """create_completion is called with logprobs=1 to enable token_logprobs."""
        branch_model.query_branch([1, 2, 3], max_tokens=3)
        _, kwargs = branch_model._llm.create_completion.call_args
        assert kwargs.get('logprobs') == 1


class TestQueryLogProbsNextToken:
    """Tests for ModelInstance.query_log_probs_next_token."""

    @pytest.fixture()
    def next_token_model(self, model_instance):
        """Configure the mock LLM to return a next-token completion response."""
        model_instance._llm.create_completion.return_value = _make_next_token_completion(
            {' Four': -0.5, ' four': -1.2}
        )
        return model_instance

    def test_returns_none_when_top_logprobs_empty(self, next_token_model) -> None:
        """Returns None when top_logprobs is empty, indicating EOS was generated."""
        next_token_model._llm.create_completion.return_value = {
            'choices': [{'logprobs': {'top_logprobs': []}, 'finish_reason': 'stop'}]
        }
        result = next_token_model.query_log_probs_next_token([1, 2, 3], n_tokens=2)
        assert result is None

    def test_returns_llmnexttokendata(self, next_token_model) -> None:
        """query_log_probs_next_token() returns an LLMNextTokenData instance."""
        result = next_token_model.query_log_probs_next_token([1, 2, 3], n_tokens=2)
        assert isinstance(result, LLMNextTokenData)

    def test_output_vec_is_passed_context(self, next_token_model) -> None:
        """output_vec on the result is the context list that was passed in."""
        context = [1, 2, 3]
        result = next_token_model.query_log_probs_next_token(context, n_tokens=2)
        assert result.output_vec == context

    def test_top_m_tokens_extracted_correctly(self, next_token_model) -> None:
        """top_m_tokens contains the dict returned by the API's top_logprobs."""
        result = next_token_model.query_log_probs_next_token([1, 2, 3], n_tokens=2)
        assert result.top_m_tokens == {' Four': -0.5, ' four': -1.2}

    def test_passes_max_tokens_one(self, next_token_model) -> None:
        """create_completion is called with max_tokens=1 to get a single next token."""
        next_token_model.query_log_probs_next_token([1, 2, 3], n_tokens=2)
        _, kwargs = next_token_model._llm.create_completion.call_args
        assert kwargs.get('max_tokens') == 1

    def test_passes_logprobs_as_n_tokens(self, next_token_model) -> None:
        """create_completion is called with logprobs set to n_tokens.

        create_completion takes logprobs as Optional[int] (the number of top
        log-probabilities to return), not a boolean as in create_chat_completion.
        """
        next_token_model.query_log_probs_next_token([1, 2, 3], n_tokens=5)
        _, kwargs = next_token_model._llm.create_completion.call_args
        assert kwargs.get('logprobs') == 5

    def test_passes_context_as_prompt(self, next_token_model) -> None:
        """create_completion receives the context list as its positional prompt argument."""
        context = [10, 20, 30]
        next_token_model.query_log_probs_next_token(context, n_tokens=2)
        args, _ = next_token_model._llm.create_completion.call_args
        assert args[0] == context


class TestFormatChatPrompt:
    """Tests for ModelInstance._format_chat_prompt."""

    @pytest.fixture()
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
        """tokenize is called on the encoded string returned by the chat handler."""
        chat_prompt_model._format_chat_prompt()
        chat_prompt_model._llm.tokenize.assert_called_once_with(
            'formatted prompt string'.encode('utf-8'),
            add_bos=False,
            special=True,
        )

    def test_returns_tokenize_output(self, chat_prompt_model) -> None:
        """The return value is whatever tokenize() returns."""
        result = chat_prompt_model._format_chat_prompt()
        assert result == [1, 2, 3, 4, 5]


@pytest.fixture()
def gen_adj_model(model_instance):
    """ModelInstance with all generate_adjusted() dependencies mocked.

    - _format_chat_prompt returns [10, 20, 30] (prompt_length = 3)
    - query_log_probs_next_token returns stable LLMNextTokenData each call
    - sample_from_logprobs (in llama_interface module) returns ' hello'
    - _llm.token_eos() returns 0 (EOS ID)
    - _llm.tokenize() returns [42] (a non-EOS token ID)
    - _llm.detokenize() returns b' hello world'
    """
    next_token_data = LLMNextTokenData(
        prompt=model_instance.context,
        output_vec=[10, 20, 30],
        top_m_tokens={' hello': -0.5, ' world': -1.2},
    )
    model_instance._llm.token_eos.return_value = 0
    model_instance._llm.tokenize.return_value = [42]
    model_instance._llm.detokenize.return_value = b' hello world'

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(model_instance, '_format_chat_prompt', return_value=[10, 20, 30])
        )
        stack.enter_context(
            patch.object(model_instance, 'query_log_probs_next_token', return_value=next_token_data)
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

    def test_returns_llmoutputdata(self, gen_adj_model) -> None:
        """generate_adjusted() returns an LLMOutputData instance."""
        result = gen_adj_model.generate_adjusted(n_tokens=2, adjust_fn=lambda ctx: ctx.token_probs, max_tokens=3)
        assert isinstance(result, LLMOutputData)

    def test_prompt_matches_context(self, gen_adj_model) -> None:
        """The prompt on the returned LLMOutputData matches the model's context."""
        result = gen_adj_model.generate_adjusted(n_tokens=2, adjust_fn=lambda ctx: ctx.token_probs, max_tokens=3)
        assert result.prompt == gen_adj_model.context

    def test_written_flag_is_false(self, gen_adj_model) -> None:
        """Freshly generated data has written=False."""
        result = gen_adj_model.generate_adjusted(n_tokens=2, adjust_fn=lambda ctx: ctx.token_probs, max_tokens=3)
        assert result.written is False

    def test_loops_exactly_max_tokens_times(self, gen_adj_model) -> None:
        """query_log_probs_next_token is called exactly max_tokens times when EOS never appears."""
        gen_adj_model.generate_adjusted(n_tokens=2, adjust_fn=lambda ctx: ctx.token_probs, max_tokens=4)
        assert gen_adj_model.query_log_probs_next_token.call_count == 4

    def test_adjust_fn_called_each_step(self, gen_adj_model) -> None:
        """adjust_fn is called once per generated token with the top_m_tokens dict."""
        adjust_fn = MagicMock(return_value={' hello': -0.5})
        gen_adj_model.generate_adjusted(n_tokens=2, adjust_fn=adjust_fn, max_tokens=3)
        assert adjust_fn.call_count == 3

    def test_adjust_fn_receives_top_m_tokens(self, gen_adj_model) -> None:
        """adjust_fn receives a GenerationContext whose token_probs is the top_m_tokens dict."""
        adjust_fn = MagicMock(return_value={' hello': -0.5})
        gen_adj_model.generate_adjusted(n_tokens=2, adjust_fn=adjust_fn, max_tokens=1)
        ctx = adjust_fn.call_args[0][0]
        assert ctx.token_probs == {' hello': -0.5, ' world': -1.2}

    def test_adjust_fn_receives_empty_prev_probs_on_first_step(self, gen_adj_model) -> None:
        """adjust_fn receives a GenerationContext with empty prev_probs on the first step."""
        adjust_fn = MagicMock(return_value={' hello': -0.5})
        gen_adj_model.generate_adjusted(n_tokens=2, adjust_fn=adjust_fn, max_tokens=1)
        ctx = adjust_fn.call_args_list[0][0][0]
        assert ctx.prev_probs == []

    def test_adjust_fn_receives_growing_prev_probs(self, gen_adj_model) -> None:
        """prev_probs grows by one entry per step, containing prob_of_token return values."""
        adjust_fn = MagicMock(return_value={' hello': -0.5})
        gen_adj_model.generate_adjusted(n_tokens=2, adjust_fn=adjust_fn, max_tokens=3)
        # Step 1: prev_probs = []; step 2: [0.8]; step 3: [0.8, 0.8]
        assert adjust_fn.call_args_list[0][0][0].prev_probs == []
        assert adjust_fn.call_args_list[1][0][0].prev_probs == [0.8]
        assert adjust_fn.call_args_list[2][0][0].prev_probs == [0.8, 0.8]

    def test_response_decoded_from_detokenize(self, gen_adj_model) -> None:
        """The response string in the result comes from decoding the generated token IDs."""
        result = gen_adj_model.generate_adjusted(n_tokens=2, adjust_fn=lambda ctx: ctx.token_probs, max_tokens=2)
        assert result.data[0] == ' hello world'

    def test_stops_when_query_returns_none(self, gen_adj_model) -> None:
        """The loop breaks immediately when query_log_probs_next_token returns None."""
        gen_adj_model.query_log_probs_next_token.return_value = None
        gen_adj_model.generate_adjusted(n_tokens=2, adjust_fn=lambda ctx: ctx.token_probs, max_tokens=10)
        assert gen_adj_model.query_log_probs_next_token.call_count == 1

    def test_stops_early_on_eos_token(self, gen_adj_model) -> None:
        """The loop breaks before max_tokens when tokenize returns the EOS token ID."""
        gen_adj_model._llm.tokenize.return_value = [0]
        gen_adj_model.generate_adjusted(n_tokens=2, adjust_fn=lambda ctx: ctx.token_probs, max_tokens=10)
        assert gen_adj_model.query_log_probs_next_token.call_count == 1

    def test_stops_early_on_empty_token_ids(self, gen_adj_model) -> None:
        """The loop breaks when tokenize returns an empty list for the sampled token."""
        gen_adj_model._llm.tokenize.return_value = []
        gen_adj_model.generate_adjusted(n_tokens=2, adjust_fn=lambda ctx: ctx.token_probs, max_tokens=10)
        assert gen_adj_model.query_log_probs_next_token.call_count == 1

    def test_prev_probs_reset_between_calls(self, gen_adj_model) -> None:
        """prev_probs starts empty on every call to generate_adjusted, not carried over."""
        adjust_fn = MagicMock(return_value={' hello': -0.5})
        gen_adj_model.generate_adjusted(n_tokens=2, adjust_fn=adjust_fn, max_tokens=2)
        gen_adj_model.query_log_probs_next_token.reset_mock()
        adjust_fn.reset_mock()
        gen_adj_model.generate_adjusted(n_tokens=2, adjust_fn=adjust_fn, max_tokens=1)
        assert adjust_fn.call_args_list[0][0][0].prev_probs == []
