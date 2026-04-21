"""Tests for ModelInstance in llama_interface.py."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from problm_solver.data import LLMOutputData


def _make_llama_mock(response_text: str = 'Mock response.') -> MagicMock:
    """Return a MagicMock that mimics the Llama chat completion API."""
    mock_llm = MagicMock()
    mock_llm.create_chat_completion.return_value = {
        'choices': [{'message': {'content': response_text}}]
    }
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
            MockLlama.return_value = MagicMock()
            ModelInstance(fname='fake.gguf', context='Hello')
            _, kwargs = MockLlama.call_args
            assert kwargs.get('logits_all') is False

    def test_logits_all_true_passed_to_llama(self) -> None:
        """logits_all=True is forwarded to the Llama constructor."""
        from problm_solver.llama_interface import ModelInstance

        with patch('problm_solver.llama_interface.Llama') as MockLlama:
            MockLlama.return_value = MagicMock()
            ModelInstance(fname='fake.gguf', context='Hello', logits_all=True)
            _, kwargs = MockLlama.call_args
            assert kwargs.get('logits_all') is True


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
