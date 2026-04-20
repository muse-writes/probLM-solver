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
