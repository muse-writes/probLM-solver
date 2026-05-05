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

    def test_correct_number_of_lines(self, sample_data: LLMOutputData, tmp_path: pytest.TempPathFactory) -> None:
        """One JSONL line is written per response."""
        out = tmp_path / 'output.jsonl'
        sample_data.write(str(out))
        lines = out.read_text(encoding='utf-8').strip().splitlines()
        assert len(lines) == len(sample_data.data)

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


@pytest.fixture()
def sample_next_token_data():
    """Module-level LLMNextTokenData fixture for write/read/roundtrip tests."""
    from problm_solver.data import LLMNextTokenData

    return LLMNextTokenData(
        prompt='What is 2+2?',
        output_vec=[1, 2, 3],
        top_m_tokens={' Four': -0.5, ' four': -1.2, ' 4': -1.8},
    )


class TestLLMNextTokenDataWrite:
    """Tests for LLMNextTokenData.write."""

    def test_creates_file(self, sample_next_token_data, tmp_path) -> None:
        """write() creates the output file."""
        out = tmp_path / 'output.json'
        sample_next_token_data.write(str(out))
        assert out.exists()

    def test_sets_written_flag(self, sample_next_token_data, tmp_path) -> None:
        """write() sets written to True."""
        out = tmp_path / 'output.json'
        sample_next_token_data.write(str(out))
        assert sample_next_token_data.written is True

    def test_prompt_matches(self, sample_next_token_data, tmp_path) -> None:
        """The prompt written to disk matches the in-memory value."""
        out = tmp_path / 'output.json'
        sample_next_token_data.write(str(out))
        assert json.loads(out.read_text(encoding='utf-8'))['prompt'] == sample_next_token_data.prompt

    def test_output_vec_matches(self, sample_next_token_data, tmp_path) -> None:
        """The output_vec written to disk matches the in-memory value."""
        out = tmp_path / 'output.json'
        sample_next_token_data.write(str(out))
        assert json.loads(out.read_text(encoding='utf-8'))['output_vec'] == sample_next_token_data.output_vec

    def test_top_m_tokens_matches(self, sample_next_token_data, tmp_path) -> None:
        """The top_m_tokens dict written to disk matches the in-memory value."""
        out = tmp_path / 'output.json'
        sample_next_token_data.write(str(out))
        record = json.loads(out.read_text(encoding='utf-8'))
        assert record['top_m_tokens'] == pytest.approx(sample_next_token_data.top_m_tokens)


class TestLLMNextTokenDataRead:
    """Tests for LLMNextTokenData.read."""

    def _make_json(self, tmp_path, prompt, output_vec, top_m_tokens) -> str:
        """Write a JSON fixture file and return its path string."""
        out = tmp_path / 'fixture.json'
        out.write_text(
            json.dumps({'prompt': prompt, 'output_vec': output_vec, 'top_m_tokens': top_m_tokens}),
            encoding='utf-8',
        )
        return str(out)

    def test_populates_prompt(self, tmp_path) -> None:
        """read() sets self.prompt from the file contents."""
        from problm_solver.data import LLMNextTokenData

        path = self._make_json(tmp_path, 'Hello?', [1, 2], {' hi': -0.3})
        obj = LLMNextTokenData(prompt='', output_vec=[], top_m_tokens={})
        obj.read(path)
        assert obj.prompt == 'Hello?'

    def test_populates_output_vec(self, tmp_path) -> None:
        """read() sets self.output_vec from the file contents."""
        from problm_solver.data import LLMNextTokenData

        path = self._make_json(tmp_path, 'Q?', [10, 20, 30], {' a': -0.5})
        obj = LLMNextTokenData(prompt='', output_vec=[], top_m_tokens={})
        obj.read(path)
        assert obj.output_vec == [10, 20, 30]

    def test_populates_top_m_tokens(self, tmp_path) -> None:
        """read() sets self.top_m_tokens from the file contents."""
        from problm_solver.data import LLMNextTokenData

        path = self._make_json(tmp_path, 'Q?', [1], {' yes': -0.1, ' no': -2.3})
        obj = LLMNextTokenData(prompt='', output_vec=[], top_m_tokens={})
        obj.read(path)
        assert obj.top_m_tokens == pytest.approx({' yes': -0.1, ' no': -2.3})

    def test_sets_written_flag(self, tmp_path) -> None:
        """read() sets written to True — the object is now in sync with disk."""
        from problm_solver.data import LLMNextTokenData

        path = self._make_json(tmp_path, 'Q?', [1, 2], {' a': -0.5})
        obj = LLMNextTokenData(prompt='', output_vec=[], top_m_tokens={})
        obj.read(path)
        assert obj.written is True


class TestLLMNextTokenDataRoundtrip:
    """Tests for LLMNextTokenData write-then-read consistency."""

    def test_roundtrip_preserves_prompt(self, sample_next_token_data, tmp_path) -> None:
        """A write-then-read cycle preserves the prompt exactly."""
        from problm_solver.data import LLMNextTokenData

        out = tmp_path / 'round.json'
        sample_next_token_data.write(str(out))
        recovered = LLMNextTokenData(prompt='', output_vec=[], top_m_tokens={})
        recovered.read(str(out))
        assert recovered.prompt == sample_next_token_data.prompt

    def test_roundtrip_preserves_output_vec(self, sample_next_token_data, tmp_path) -> None:
        """A write-then-read cycle preserves output_vec exactly."""
        from problm_solver.data import LLMNextTokenData

        out = tmp_path / 'round.json'
        sample_next_token_data.write(str(out))
        recovered = LLMNextTokenData(prompt='', output_vec=[], top_m_tokens={})
        recovered.read(str(out))
        assert recovered.output_vec == sample_next_token_data.output_vec

    def test_roundtrip_preserves_top_m_tokens(self, sample_next_token_data, tmp_path) -> None:
        """A write-then-read cycle preserves top_m_tokens exactly."""
        from problm_solver.data import LLMNextTokenData

        out = tmp_path / 'round.json'
        sample_next_token_data.write(str(out))
        recovered = LLMNextTokenData(prompt='', output_vec=[], top_m_tokens={})
        recovered.read(str(out))
        assert recovered.top_m_tokens == pytest.approx(sample_next_token_data.top_m_tokens)


class TestLLMTokenDataRoundtrip:
    """Tests for LLMTokenData write-then-read consistency."""

    def test_roundtrip_preserves_prompt(self, sample_token_data, tmp_path) -> None:
        """A write-then-read cycle preserves the prompt exactly."""
        from problm_solver.data import LLMTokenData

        out = tmp_path / 'round.json'
        sample_token_data.write(str(out))
        recovered = LLMTokenData(prompt='', tokens=[], probs=[])
        recovered.read(str(out))
        assert recovered.prompt == sample_token_data.prompt

    def test_roundtrip_preserves_tokens(self, sample_token_data, tmp_path) -> None:
        """A write-then-read cycle preserves the token list exactly."""
        from problm_solver.data import LLMTokenData

        out = tmp_path / 'round.json'
        sample_token_data.write(str(out))
        recovered = LLMTokenData(prompt='', tokens=[], probs=[])
        recovered.read(str(out))
        assert recovered.tokens == sample_token_data.tokens

    def test_roundtrip_preserves_probs(self, sample_token_data, tmp_path) -> None:
        """A write-then-read cycle preserves all probabilities exactly."""
        from problm_solver.data import LLMTokenData

        out = tmp_path / 'round.json'
        sample_token_data.write(str(out))
        recovered = LLMTokenData(prompt='', tokens=[], probs=[])
        recovered.read(str(out))
        assert recovered.probs == pytest.approx(sample_token_data.probs)


# ---------------------------------------------------------------------------
# Hyperparams + LLMOutputDataFull
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_hyperparams():
    """Return a Hyperparams instance for reuse across LLMOutputDataFull tests."""
    from problm_solver.data import Hyperparams

    return Hyperparams(alpha=0.7, top_k=10, top_p=None, max_tokens=50)


@pytest.fixture()
def sample_full_data(sample_hyperparams):
    """Return an LLMOutputDataFull instance for reuse across write/read/roundtrip tests."""
    from problm_solver.data import LLMOutputDataFull

    return LLMOutputDataFull(
        context=['Hello', ',', ' world'],
        hyperparams=sample_hyperparams,
        response_probabilities={' Hello': 0.7, ' world': 0.3},
        response_topk=(
            ['Hello', 'world'],
            [{' Hello': -0.3, ' Hi': -1.2}, {' world': -0.5, ' there': -0.8}],
        ),
        sampling_method='low_temp',
        branch_sampler=None,
    )


class TestLLMOutputDataFullWrite:
    """Tests for LLMOutputDataFull.write."""

    def test_creates_file(self, sample_full_data, tmp_path) -> None:
        """write() creates the output file."""
        out = tmp_path / 'output.json'
        sample_full_data.write(str(out))
        assert out.exists()

    def test_sets_written_flag(self, sample_full_data, tmp_path) -> None:
        """write() sets _written to True."""
        out = tmp_path / 'output.json'
        sample_full_data.write(str(out))
        assert sample_full_data._written is True

    def test_written_flag_false_before_write(self, sample_full_data) -> None:
        """_written is False before write() is called."""
        assert sample_full_data._written is False


class TestLLMOutputDataFullRead:
    """Tests for LLMOutputDataFull.read."""

    def _make_json(self, tmp_path, *, context, hyperparams_dict, response_probabilities,
                   response_topk, sampling_method, branch_sampler) -> str:
        """Write a JSON fixture file and return its path string."""
        out = tmp_path / 'fixture.json'
        record = {
            'context': context,
            'hyperparams': hyperparams_dict,
            'response_probabilities': response_probabilities,
            'response_topk': response_topk,
            'sampling_method': sampling_method,
            'branch_sampler': branch_sampler,
        }
        out.write_text(json.dumps(record), encoding='utf-8')
        return str(out)

    @pytest.fixture()
    def _fixture_path(self, tmp_path):
        """Pre-built fixture JSON path for read tests."""
        return self._make_json(
            tmp_path,
            context=['Hi', '!'],
            hyperparams_dict={'alpha': 0.5, 'top_k': 5, 'top_p': None, 'max_tokens': 20},
            response_probabilities={' Hi': 0.8, ' Hello': 0.2},
            response_topk=[
                ['Hi', 'Hello'],
                [{' Hi': -0.2, ' Hey': -0.9}, {' Hello': -0.5, ' World': -1.1}],
            ],
            sampling_method='power_mcmc',
            branch_sampler='metropolis',
        )

    def _blank_obj(self):
        """Return an empty LLMOutputDataFull suitable for reading into."""
        from problm_solver.data import Hyperparams, LLMOutputDataFull

        return LLMOutputDataFull(
            context=[],
            hyperparams=Hyperparams(alpha=0.0, top_k=None, top_p=None, max_tokens=0),
            response_probabilities={},
            response_topk=([], []),
            sampling_method='',
            branch_sampler=None,
        )

    def test_sets_written_flag(self, _fixture_path) -> None:
        """read() sets _written to True — the object is now in sync with disk."""
        obj = self._blank_obj()
        obj.read(_fixture_path)
        assert obj._written is True


class TestLLMOutputDataFullRoundtrip:
    """Tests for LLMOutputDataFull write-then-read consistency."""

    def _blank_obj(self):
        """Return an empty LLMOutputDataFull suitable for reading into."""
        from problm_solver.data import Hyperparams, LLMOutputDataFull

        return LLMOutputDataFull(
            context=[],
            hyperparams=Hyperparams(alpha=0.0, top_k=None, top_p=None, max_tokens=0),
            response_probabilities={},
            response_topk=([], []),
            sampling_method='',
            branch_sampler=None,
        )

    def test_roundtrip_preserves_context(self, sample_full_data, tmp_path) -> None:
        """A write-then-read cycle preserves context exactly."""
        out = tmp_path / 'round.json'
        sample_full_data.write(str(out))
        recovered = self._blank_obj()
        recovered.read(str(out))
        assert recovered.context == sample_full_data.context

    def test_roundtrip_preserves_hyperparams_alpha(self, sample_full_data, tmp_path) -> None:
        """A write-then-read cycle preserves hyperparams.alpha."""
        out = tmp_path / 'round.json'
        sample_full_data.write(str(out))
        recovered = self._blank_obj()
        recovered.read(str(out))
        assert recovered.hyperparams.alpha == pytest.approx(sample_full_data.hyperparams.alpha)

    def test_roundtrip_preserves_hyperparams_fields(self, sample_full_data, tmp_path) -> None:
        """A write-then-read cycle preserves all Hyperparams fields."""
        out = tmp_path / 'round.json'
        sample_full_data.write(str(out))
        recovered = self._blank_obj()
        recovered.read(str(out))
        assert recovered.hyperparams.top_k == sample_full_data.hyperparams.top_k
        assert recovered.hyperparams.top_p == sample_full_data.hyperparams.top_p
        assert recovered.hyperparams.max_tokens == sample_full_data.hyperparams.max_tokens

    def test_roundtrip_preserves_response_probabilities(self, sample_full_data, tmp_path) -> None:
        """A write-then-read cycle preserves response_probabilities."""
        out = tmp_path / 'round.json'
        sample_full_data.write(str(out))
        recovered = self._blank_obj()
        recovered.read(str(out))
        assert recovered.response_probabilities == pytest.approx(
            sample_full_data.response_probabilities
        )

    def test_roundtrip_preserves_response_topk_tokens(self, sample_full_data, tmp_path) -> None:
        """A write-then-read cycle preserves the token list inside response_topk."""
        out = tmp_path / 'round.json'
        sample_full_data.write(str(out))
        recovered = self._blank_obj()
        recovered.read(str(out))
        orig_tokens = sample_full_data.response_topk[0]
        assert list(recovered.response_topk[0]) == orig_tokens

    def test_roundtrip_preserves_response_topk_probs(self, sample_full_data, tmp_path) -> None:
        """A write-then-read cycle preserves the per-token top-k dicts inside response_topk."""
        out = tmp_path / 'round.json'
        sample_full_data.write(str(out))
        recovered = self._blank_obj()
        recovered.read(str(out))
        orig_topk = sample_full_data.response_topk[1]
        for recovered_dict, orig_dict in zip(recovered.response_topk[1], orig_topk):
            assert dict(recovered_dict) == pytest.approx(orig_dict)

    def test_roundtrip_preserves_sampling_method(self, sample_full_data, tmp_path) -> None:
        """A write-then-read cycle preserves sampling_method exactly."""
        out = tmp_path / 'round.json'
        sample_full_data.write(str(out))
        recovered = self._blank_obj()
        recovered.read(str(out))
        assert recovered.sampling_method == sample_full_data.sampling_method

    def test_roundtrip_preserves_branch_sampler(self, sample_full_data, tmp_path) -> None:
        """A write-then-read cycle preserves branch_sampler (including None)."""
        out = tmp_path / 'round.json'
        sample_full_data.write(str(out))
        recovered = self._blank_obj()
        recovered.read(str(out))
        assert recovered.branch_sampler == sample_full_data.branch_sampler
