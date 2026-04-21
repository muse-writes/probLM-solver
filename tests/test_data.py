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


class TestLLMTokenDataInit:
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

    def test_written_is_false_on_init(self, sample_token_data) -> None:
        """written flag starts as False — no disk I/O has occurred yet."""
        assert sample_token_data.written is False


class TestLLMOutputDataInit:
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


@pytest.fixture()
def sample_token_data():
    """Return a small LLMTokenData instance for reuse across token data test classes."""
    from problm_solver.data import LLMTokenData

    return LLMTokenData(
        prompt='What is 2+2?',
        tokens=['Four', '.'],
        probs=[0.9, 0.99],
    )


class TestTokenProbError:
    """Tests for the TokenProbError exception."""

    def test_is_value_error(self) -> None:
        """TokenProbError is a subclass of ValueError."""
        from problm_solver.data import TokenProbError

        assert issubclass(TokenProbError, ValueError)

    def test_raised_on_mismatched_lengths(self) -> None:
        """LLMTokenData raises TokenProbError when tokens and probs differ in length."""
        from problm_solver.data import LLMTokenData, TokenProbError

        with pytest.raises(TokenProbError):
            LLMTokenData(prompt='Q?', tokens=['a', 'b'], probs=[0.9])

    def test_not_raised_on_equal_lengths(self) -> None:
        """LLMTokenData does not raise when tokens and probs have equal length."""
        from problm_solver.data import LLMTokenData

        LLMTokenData(prompt='Q?', tokens=['a'], probs=[0.9])  # must not raise

    def test_not_raised_on_empty(self) -> None:
        """LLMTokenData does not raise when both tokens and probs are empty."""
        from problm_solver.data import LLMTokenData

        LLMTokenData(prompt='Q?', tokens=[], probs=[])  # must not raise

    def test_error_message_mentions_class(self) -> None:
        """The error message includes the class name for easier debugging."""
        from problm_solver.data import LLMTokenData, TokenProbError

        try:
            LLMTokenData(prompt='Q?', tokens=['a'], probs=[0.9, 0.8])
        except TokenProbError as exc:
            assert 'LLMTokenData' in str(exc)


class TestLLMTokenDataWrite:
    """Tests for LLMTokenData.write."""

    def test_creates_file(self, sample_token_data, tmp_path: pytest.TempPathFactory) -> None:
        """write() creates the output file."""
        out = tmp_path / 'output.json'
        sample_token_data.write(str(out))
        assert out.exists()

    def test_sets_written_flag(self, sample_token_data, tmp_path: pytest.TempPathFactory) -> None:
        """write() sets written to True."""
        out = tmp_path / 'output.json'
        sample_token_data.write(str(out))
        assert sample_token_data.written is True

    def test_written_flag_false_before_write(self, sample_token_data) -> None:
        """written is False before write() is called."""
        assert sample_token_data.written is False

    def test_output_is_valid_json(self, sample_token_data, tmp_path: pytest.TempPathFactory) -> None:
        """The output file contains valid JSON."""
        out = tmp_path / 'output.json'
        sample_token_data.write(str(out))
        json.loads(out.read_text(encoding='utf-8'))  # raises if invalid

    def test_record_has_prompt_field(self, sample_token_data, tmp_path: pytest.TempPathFactory) -> None:
        """The JSON record contains a 'prompt' key."""
        out = tmp_path / 'output.json'
        sample_token_data.write(str(out))
        record = json.loads(out.read_text(encoding='utf-8'))
        assert 'prompt' in record

    def test_record_has_tokens_field(self, sample_token_data, tmp_path: pytest.TempPathFactory) -> None:
        """The JSON record contains a 'tokens' key."""
        out = tmp_path / 'output.json'
        sample_token_data.write(str(out))
        record = json.loads(out.read_text(encoding='utf-8'))
        assert 'tokens' in record

    def test_record_has_probs_field(self, sample_token_data, tmp_path: pytest.TempPathFactory) -> None:
        """The JSON record contains a 'probs' key."""
        out = tmp_path / 'output.json'
        sample_token_data.write(str(out))
        record = json.loads(out.read_text(encoding='utf-8'))
        assert 'probs' in record

    def test_prompt_matches(self, sample_token_data, tmp_path: pytest.TempPathFactory) -> None:
        """The prompt written to disk matches the in-memory value."""
        out = tmp_path / 'output.json'
        sample_token_data.write(str(out))
        record = json.loads(out.read_text(encoding='utf-8'))
        assert record['prompt'] == sample_token_data.prompt

    def test_tokens_match(self, sample_token_data, tmp_path: pytest.TempPathFactory) -> None:
        """The token list written to disk matches the in-memory value."""
        out = tmp_path / 'output.json'
        sample_token_data.write(str(out))
        record = json.loads(out.read_text(encoding='utf-8'))
        assert record['tokens'] == sample_token_data.tokens

    def test_probs_match(self, sample_token_data, tmp_path: pytest.TempPathFactory) -> None:
        """The probability list written to disk matches the in-memory value."""
        out = tmp_path / 'output.json'
        sample_token_data.write(str(out))
        record = json.loads(out.read_text(encoding='utf-8'))
        assert record['probs'] == pytest.approx(sample_token_data.probs)


class TestLLMTokenDataRead:
    """Tests for LLMTokenData.read."""

    def _make_json(self, tmp_path, prompt: str, tokens: list, probs: list) -> str:
        """Write a JSON fixture file and return its path string."""
        out = tmp_path / 'fixture.json'
        out.write_text(
            json.dumps({'prompt': prompt, 'tokens': tokens, 'probs': probs}),
            encoding='utf-8',
        )
        return str(out)

    def test_populates_prompt(self, tmp_path: pytest.TempPathFactory) -> None:
        """read() sets self.prompt from the file contents."""
        from problm_solver.data import LLMTokenData

        path = self._make_json(tmp_path, 'Hello?', ['Hi', '!'], [0.8, 0.99])
        obj = LLMTokenData(prompt='', tokens=[], probs=[])
        obj.read(path)
        assert obj.prompt == 'Hello?'

    def test_populates_tokens(self, tmp_path: pytest.TempPathFactory) -> None:
        """read() sets self.tokens from the file contents."""
        from problm_solver.data import LLMTokenData

        path = self._make_json(tmp_path, 'Q?', ['Four', '.'], [0.9, 0.99])
        obj = LLMTokenData(prompt='', tokens=[], probs=[])
        obj.read(path)
        assert obj.tokens == ['Four', '.']

    def test_populates_probs(self, tmp_path: pytest.TempPathFactory) -> None:
        """read() sets self.probs from the file contents."""
        from problm_solver.data import LLMTokenData

        path = self._make_json(tmp_path, 'Q?', ['Four', '.'], [0.9, 0.99])
        obj = LLMTokenData(prompt='', tokens=[], probs=[])
        obj.read(path)
        assert obj.probs == pytest.approx([0.9, 0.99])

    def test_sets_written_flag(self, tmp_path: pytest.TempPathFactory) -> None:
        """read() sets written to True."""
        from problm_solver.data import LLMTokenData

        path = self._make_json(tmp_path, 'Q?', ['a'], [0.5])
        obj = LLMTokenData(prompt='', tokens=[], probs=[])
        obj.read(path)
        assert obj.written is True


class TestLLMTokenDataRoundtrip:
    """Tests for LLMTokenData write-then-read consistency."""

    def test_roundtrip_preserves_prompt(self, sample_token_data, tmp_path: pytest.TempPathFactory) -> None:
        """A write-then-read cycle preserves the prompt exactly."""
        from problm_solver.data import LLMTokenData

        out = tmp_path / 'round.json'
        sample_token_data.write(str(out))
        recovered = LLMTokenData(prompt='', tokens=[], probs=[])
        recovered.read(str(out))
        assert recovered.prompt == sample_token_data.prompt

    def test_roundtrip_preserves_tokens(self, sample_token_data, tmp_path: pytest.TempPathFactory) -> None:
        """A write-then-read cycle preserves the token list exactly."""
        from problm_solver.data import LLMTokenData

        out = tmp_path / 'round.json'
        sample_token_data.write(str(out))
        recovered = LLMTokenData(prompt='', tokens=[], probs=[])
        recovered.read(str(out))
        assert recovered.tokens == sample_token_data.tokens

    def test_roundtrip_preserves_probs(self, sample_token_data, tmp_path: pytest.TempPathFactory) -> None:
        """A write-then-read cycle preserves all probabilities exactly."""
        from problm_solver.data import LLMTokenData

        out = tmp_path / 'round.json'
        sample_token_data.write(str(out))
        recovered = LLMTokenData(prompt='', tokens=[], probs=[])
        recovered.read(str(out))
        assert recovered.probs == pytest.approx(sample_token_data.probs)
