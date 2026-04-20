"""Tests for LLMOutputData in data.py."""

import json

import numpy as np
import pytest

from problm_solver.data import LLMOutputData


@pytest.fixture()
def sample_data() -> LLMOutputData:
    """Return a small LLMOutputData instance for reuse across tests."""
    return LLMOutputData(
        prompt='What is 2 + 2?',
        data=np.array(['Four.', 'It is 4.', '2 + 2 equals 4.'], dtype=str),
    )


class TestLLMTokenData:
    """Tests for LLMTokenData.__init__."""

    @pytest.fixture()
    def sample_token_data(self):
        """Return a small LLMTokenData instance for reuse."""
        from problm_solver.data import LLMTokenData

        return LLMTokenData(
            prompt='What is 2+2?',
            tokens=['Four', '.'],
            probs=[0.9, 0.99],
        )

    def test_stores_prompt(self, sample_token_data) -> None:
        """prompt attribute is set correctly on construction."""
        assert sample_token_data.prompt == 'What is 2+2?'

    def test_stores_tokens(self, sample_token_data) -> None:
        """tokens attribute is set correctly on construction."""
        assert sample_token_data.tokens == ['Four', '.']

    def test_stores_probs(self, sample_token_data) -> None:
        """probs attribute is set correctly on construction."""
        assert sample_token_data.probs == [0.9, 0.99]

    def test_tokens_and_probs_aligned(self, sample_token_data) -> None:
        """tokens and probs have equal length and are positionally aligned."""
        assert len(sample_token_data.tokens) == len(sample_token_data.probs)



    """Tests for LLMOutputData.__init__."""

    def test_stores_prompt(self, sample_data: LLMOutputData) -> None:
        """Prompt attribute is set correctly on construction."""
        assert sample_data.prompt == 'What is 2 + 2?'

    def test_stores_data(self, sample_data: LLMOutputData) -> None:
        """Data attribute is set correctly on construction."""
        assert list(sample_data.data) == ['Four.', 'It is 4.', '2 + 2 equals 4.']

    def test_written_is_false_on_init(self, sample_data: LLMOutputData) -> None:
        """written flag starts as False — no disk I/O has occurred yet."""
        assert sample_data.written is False


class TestLLMOutputDataWrite:
    """Tests for LLMOutputData.write."""

    def test_creates_file(self, sample_data: LLMOutputData, tmp_path: pytest.TempPathFactory) -> None:
        """write() creates the output file."""
        out = tmp_path / 'output.jsonl'
        sample_data.write(str(out))
        assert out.exists()

    def test_sets_written_flag(self, sample_data: LLMOutputData, tmp_path: pytest.TempPathFactory) -> None:
        """write() sets written to True."""
        out = tmp_path / 'output.jsonl'
        sample_data.write(str(out))
        assert sample_data.written is True

    def test_output_is_valid_jsonl(self, sample_data: LLMOutputData, tmp_path: pytest.TempPathFactory) -> None:
        """Every line of the output file is valid JSON."""
        out = tmp_path / 'output.jsonl'
        sample_data.write(str(out))
        lines = out.read_text(encoding='utf-8').strip().splitlines()
        for line in lines:
            json.loads(line)  # raises if invalid

    def test_correct_number_of_lines(self, sample_data: LLMOutputData, tmp_path: pytest.TempPathFactory) -> None:
        """One JSONL line is written per response."""
        out = tmp_path / 'output.jsonl'
        sample_data.write(str(out))
        lines = out.read_text(encoding='utf-8').strip().splitlines()
        assert len(lines) == len(sample_data.data)

    def test_record_fields_present(self, sample_data: LLMOutputData, tmp_path: pytest.TempPathFactory) -> None:
        """Each record contains id, prompt, and response keys."""
        out = tmp_path / 'output.jsonl'
        sample_data.write(str(out))
        lines = out.read_text(encoding='utf-8').strip().splitlines()
        for line in lines:
            record = json.loads(line)
            assert 'id' in record
            assert 'prompt' in record
            assert 'response' in record

    def test_ids_start_at_one_and_increment(self, sample_data: LLMOutputData, tmp_path: pytest.TempPathFactory) -> None:
        """Record IDs run 1, 2, 3, ... (1-indexed, not 0-indexed)."""
        out = tmp_path / 'output.jsonl'
        sample_data.write(str(out))
        lines = out.read_text(encoding='utf-8').strip().splitlines()
        ids = [json.loads(line)['id'] for line in lines]
        assert ids == list(range(1, len(sample_data.data) + 1))

    def test_prompt_in_every_record(self, sample_data: LLMOutputData, tmp_path: pytest.TempPathFactory) -> None:
        """The prompt is recorded in every JSONL line."""
        out = tmp_path / 'output.jsonl'
        sample_data.write(str(out))
        lines = out.read_text(encoding='utf-8').strip().splitlines()
        for line in lines:
            assert json.loads(line)['prompt'] == sample_data.prompt

    def test_responses_match_data_array(self, sample_data: LLMOutputData, tmp_path: pytest.TempPathFactory) -> None:
        """Responses written to disk match the in-memory data array."""
        out = tmp_path / 'output.jsonl'
        sample_data.write(str(out))
        lines = out.read_text(encoding='utf-8').strip().splitlines()
        written_responses = [json.loads(line)['response'] for line in lines]
        assert written_responses == list(sample_data.data)


class TestLLMOutputDataRead:
    """Tests for LLMOutputData.read."""

    def _make_jsonl(self, tmp_path: pytest.TempPathFactory, prompt: str, responses: list[str]) -> str:
        """Write a JSONL fixture file and return its path string."""
        out = tmp_path / 'fixture.jsonl'
        with out.open('w', encoding='utf-8') as f:
            for i, resp in enumerate(responses, start=1):
                f.write(json.dumps({'id': i, 'prompt': prompt, 'response': resp}) + '\n')
        return str(out)

    def test_populates_prompt(self, tmp_path: pytest.TempPathFactory) -> None:
        """read() sets self.prompt from the file contents."""
        path = self._make_jsonl(tmp_path, 'Hello?', ['Hi!', 'Hey!'])
        obj = LLMOutputData(prompt='', data=np.array([], dtype=str))
        obj.read(path)
        assert obj.prompt == 'Hello?'

    def test_populates_data(self, tmp_path: pytest.TempPathFactory) -> None:
        """read() fills self.data with the responses from the file."""
        responses = ['Response A', 'Response B', 'Response C']
        path = self._make_jsonl(tmp_path, 'Prompt?', responses)
        obj = LLMOutputData(prompt='', data=np.array([], dtype=str))
        obj.read(path)
        assert list(obj.data) == responses

    def test_data_is_numpy_array(self, tmp_path: pytest.TempPathFactory) -> None:
        """read() stores data as a numpy array."""
        path = self._make_jsonl(tmp_path, 'Q?', ['A', 'B'])
        obj = LLMOutputData(prompt='', data=np.array([], dtype=str))
        obj.read(path)
        assert isinstance(obj.data, np.ndarray)

    def test_sets_written_flag(self, tmp_path: pytest.TempPathFactory) -> None:
        """read() sets written to True — the object is now in sync with disk."""
        path = self._make_jsonl(tmp_path, 'Q?', ['A'])
        obj = LLMOutputData(prompt='', data=np.array([], dtype=str))
        obj.read(path)
        assert obj.written is True


class TestLLMOutputDataRoundtrip:
    """Tests for write-then-read consistency."""

    def test_roundtrip_preserves_prompt(self, sample_data: LLMOutputData, tmp_path: pytest.TempPathFactory) -> None:
        """A write-then-read cycle preserves the prompt exactly."""
        out = tmp_path / 'round.jsonl'
        sample_data.write(str(out))
        recovered = LLMOutputData(prompt='', data=np.array([], dtype=str))
        recovered.read(str(out))
        assert recovered.prompt == sample_data.prompt

    def test_roundtrip_preserves_responses(self, sample_data: LLMOutputData, tmp_path: pytest.TempPathFactory) -> None:
        """A write-then-read cycle preserves all responses exactly."""
        out = tmp_path / 'round.jsonl'
        sample_data.write(str(out))
        recovered = LLMOutputData(prompt='', data=np.array([], dtype=str))
        recovered.read(str(out))
        assert list(recovered.data) == list(sample_data.data)
