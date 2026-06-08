"""Tests for data containers in data.py."""

import json

import numpy as np
import pytest

from problm_solver.data import (
    Hyperparams,
    LLMNextTokenData,
    LLMOutputData,
    LLMOutputDataFull,
    LLMTokenData,
    TokenProbError,
)


@pytest.fixture()
def sample_output_data() -> LLMOutputData:
    """Return a small LLMOutputData instance for reuse across tests."""
    return LLMOutputData(
        prompt='What is 2 + 2?',
        data=np.array(['Four.', 'It is 4.', '2 + 2 equals 4.'], dtype=str),
    )


class TestLLMOutputData:
    """Tests for LLMOutputData write/read/roundtrip behavior."""

    def test_write_outputs_valid_jsonl_and_sets_written(
        self,
        sample_output_data: LLMOutputData,
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        """write() writes one valid JSON object per line and flips written=True."""
        out = tmp_path / 'output.jsonl'
        sample_output_data.write(str(out))

        assert out.exists()
        assert sample_output_data.written is True

        records = [json.loads(line) for line in out.read_text(encoding='utf-8').splitlines()]
        assert len(records) == len(sample_output_data.data)
        assert [r['id'] for r in records] == [1, 2, 3]
        assert all(r['prompt'] == sample_output_data.prompt for r in records)
        assert [r['response'] for r in records] == list(sample_output_data.data)

    def test_read_populates_fields_and_sets_written(self, tmp_path: pytest.TempPathFactory) -> None:
        """read() restores prompt/data from JSONL and sets written=True."""
        out = tmp_path / 'fixture.jsonl'
        lines = [
            {'id': 1, 'prompt': 'Hello?', 'response': 'Hi!'},
            {'id': 2, 'prompt': 'Hello?', 'response': 'Hey!'},
        ]
        out.write_text('\n'.join(json.dumps(x) for x in lines) + '\n', encoding='utf-8')

        obj = LLMOutputData(prompt='', data=np.array([], dtype=str))
        obj.read(str(out))

        assert obj.prompt == 'Hello?'
        assert isinstance(obj.data, np.ndarray)
        assert list(obj.data) == ['Hi!', 'Hey!']
        assert obj.written is True

    def test_roundtrip_preserves_prompt_and_responses(
        self,
        sample_output_data: LLMOutputData,
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        """A write-then-read cycle preserves prompt and responses exactly."""
        out = tmp_path / 'round.jsonl'
        sample_output_data.write(str(out))

        recovered = LLMOutputData(prompt='', data=np.array([], dtype=str))
        recovered.read(str(out))

        assert recovered.prompt == sample_output_data.prompt
        assert list(recovered.data) == list(sample_output_data.data)


class TestTokenProbError:
    """Tests for the TokenProbError exception."""

    def test_is_value_error(self) -> None:
        """TokenProbError is a ValueError subclass."""
        assert issubclass(TokenProbError, ValueError)

    def test_raised_on_mismatched_lengths_and_message(self) -> None:
        """LLMTokenData raises TokenProbError with a useful class-name message."""
        with pytest.raises(TokenProbError) as exc:
            LLMTokenData(prompt='Q?', tokens=['a', 'b'], probs=[0.9])
        assert 'LLMTokenData' in str(exc.value)

    def test_not_raised_when_lengths_match_or_empty(self) -> None:
        """Equal-length and empty token/prob arrays are accepted."""
        LLMTokenData(prompt='Q?', tokens=['a'], probs=[0.9])
        LLMTokenData(prompt='Q?', tokens=[], probs=[])


@pytest.fixture()
def sample_token_data() -> LLMTokenData:
    """Return a small LLMTokenData instance for reuse across tests."""
    return LLMTokenData(
        prompt='What is 2+2?',
        tokens=['Four', '.'],
        probs=[0.9, 0.99],
    )


class TestLLMTokenData:
    """Tests for LLMTokenData write/read/roundtrip behavior."""

    def test_write_outputs_valid_json_and_sets_written(
        self,
        sample_token_data: LLMTokenData,
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        """write() writes expected JSON fields and sets written=True."""
        out = tmp_path / 'output.json'
        assert sample_token_data.written is False

        sample_token_data.write(str(out))
        record = json.loads(out.read_text(encoding='utf-8'))

        assert out.exists()
        assert sample_token_data.written is True
        assert record['prompt'] == sample_token_data.prompt
        assert record['tokens'] == sample_token_data.tokens
        assert record['probs'] == pytest.approx(sample_token_data.probs)

    def test_read_populates_fields_and_sets_written(self, tmp_path: pytest.TempPathFactory) -> None:
        """read() restores prompt/tokens/probs and sets written=True."""
        out = tmp_path / 'fixture.json'
        out.write_text(
            json.dumps({'prompt': 'Hello?', 'tokens': ['Hi', '!'], 'probs': [0.8, 0.99]}),
            encoding='utf-8',
        )

        obj = LLMTokenData(prompt='', tokens=[], probs=[])
        obj.read(str(out))

        assert obj.prompt == 'Hello?'
        assert obj.tokens == ['Hi', '!']
        assert obj.probs == pytest.approx([0.8, 0.99])
        assert obj.written is True

    def test_roundtrip_preserves_prompt_tokens_and_probs(
        self,
        sample_token_data: LLMTokenData,
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        """A write-then-read cycle preserves all fields exactly."""
        out = tmp_path / 'round.json'
        sample_token_data.write(str(out))

        recovered = LLMTokenData(prompt='', tokens=[], probs=[])
        recovered.read(str(out))

        assert recovered.prompt == sample_token_data.prompt
        assert recovered.tokens == sample_token_data.tokens
        assert recovered.probs == pytest.approx(sample_token_data.probs)


@pytest.fixture()
def sample_next_token_data() -> LLMNextTokenData:
    """Return a small LLMNextTokenData instance for reuse across tests."""
    return LLMNextTokenData(
        prompt='What is 2+2?',
        output_vec=[1, 2, 3],
        top_k_tokens={' Four': -0.5, ' four': -1.2, ' 4': -1.8},
    )


class TestLLMNextTokenData:
    """Tests for LLMNextTokenData write/read/roundtrip behavior."""

    def test_write_outputs_valid_json_and_sets_written(
        self,
        sample_next_token_data: LLMNextTokenData,
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        """write() writes expected JSON fields and sets written=True."""
        out = tmp_path / 'output.json'
        sample_next_token_data.write(str(out))
        record = json.loads(out.read_text(encoding='utf-8'))

        assert out.exists()
        assert sample_next_token_data.written is True
        assert record['prompt'] == sample_next_token_data.prompt
        assert record['output_vec'] == sample_next_token_data.output_vec
        assert record['top_k_tokens'] == pytest.approx(sample_next_token_data.top_k_tokens)

    def test_read_populates_fields_and_sets_written(self, tmp_path: pytest.TempPathFactory) -> None:
        """read() restores prompt/output_vec/top_k_tokens and sets written=True."""
        out = tmp_path / 'fixture.json'
        out.write_text(
            json.dumps({'prompt': 'Q?', 'output_vec': [10, 20], 'top_k_tokens': {' yes': -0.1}}),
            encoding='utf-8',
        )

        obj = LLMNextTokenData(prompt='', output_vec=[], top_k_tokens={})
        obj.read(str(out))

        assert obj.prompt == 'Q?'
        assert obj.output_vec == [10, 20]
        assert obj.top_k_tokens == pytest.approx({' yes': -0.1})
        assert obj.written is True

    def test_roundtrip_preserves_prompt_output_vec_and_top_k(
        self,
        sample_next_token_data: LLMNextTokenData,
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        """A write-then-read cycle preserves all fields exactly."""
        out = tmp_path / 'round.json'
        sample_next_token_data.write(str(out))

        recovered = LLMNextTokenData(prompt='', output_vec=[], top_k_tokens={})
        recovered.read(str(out))

        assert recovered.prompt == sample_next_token_data.prompt
        assert recovered.output_vec == sample_next_token_data.output_vec
        assert recovered.top_k_tokens == pytest.approx(sample_next_token_data.top_k_tokens)


@pytest.fixture()
def sample_hyperparams() -> Hyperparams:
    """Return Hyperparams for LLMOutputDataFull tests."""
    return Hyperparams(alpha=0.7, top_k=10, top_p=None, max_tokens=50)


@pytest.fixture()
def sample_full_data(sample_hyperparams: Hyperparams) -> LLMOutputDataFull:
    """Return an LLMOutputDataFull instance for write/read/roundtrip tests."""
    return LLMOutputDataFull(
        context=['Hello', ',', ' world'],
        hyperparams=sample_hyperparams,
        response_probabilities=([' Hello', ' world'], [0.7, 0.3]),
        response_topk=(
            ['Hello', 'world'],
            [{' Hello': -0.3, ' Hi': -1.2}, {' world': -0.5, ' there': -0.8}],
        ),
        sampling_method='low_temp',
        branch_sampler=None,
    )


class TestLLMOutputDataFull:
    """Tests for LLMOutputDataFull write/read/roundtrip behavior."""

    @staticmethod
    def _blank_obj() -> LLMOutputDataFull:
        """Return an empty LLMOutputDataFull suitable for reading into."""
        return LLMOutputDataFull(
            context=[],
            hyperparams=Hyperparams(alpha=0.0, top_k=None, top_p=None, max_tokens=0),
            response_probabilities=([], []),
            response_topk=([], []),
            sampling_method='',
            branch_sampler=None,
        )

    def test_write_outputs_valid_json_and_sets_written(
        self,
        sample_full_data: LLMOutputDataFull,
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        """write() serializes expected fields (excluding _written) and sets _written=True."""
        out = tmp_path / 'output.json'
        assert sample_full_data._written is False

        sample_full_data.write(str(out))
        record = json.loads(out.read_text(encoding='utf-8'))

        assert out.exists()
        assert sample_full_data._written is True
        assert '_written' not in record
        assert record['context'] == sample_full_data.context
        assert record['hyperparams'] == {
            'alpha': sample_full_data.hyperparams.alpha,
            'top_k': sample_full_data.hyperparams.top_k,
            'top_p': sample_full_data.hyperparams.top_p,
            'max_tokens': sample_full_data.hyperparams.max_tokens,
        }
        assert record['sampling_method'] == sample_full_data.sampling_method
        assert record['branch_sampler'] == sample_full_data.branch_sampler

    def test_read_populates_fields_and_sets_written(self, tmp_path: pytest.TempPathFactory) -> None:
        """read() restores all top-level fields and sets _written=True."""
        out = tmp_path / 'fixture.json'
        out.write_text(
            json.dumps({
                'context': ['Hi', '!'],
                'hyperparams': {'alpha': 0.5, 'top_k': 5, 'top_p': None, 'max_tokens': 20},
                'response_probabilities': [[' Hi', ' Hello'], [0.8, 0.2]],
                'response_topk': [
                    ['Hi', 'Hello'],
                    [{' Hi': -0.2, ' Hey': -0.9}, {' Hello': -0.5, ' World': -1.1}],
                ],
                'sampling_method': 'power_mcmc',
                'branch_sampler': 'metropolis',
            }),
            encoding='utf-8',
        )

        obj = self._blank_obj()
        obj.read(str(out))

        assert obj.context == ['Hi', '!']
        assert isinstance(obj.hyperparams, Hyperparams)
        assert obj.hyperparams.alpha == pytest.approx(0.5)
        assert obj.hyperparams.top_k == 5
        assert obj.hyperparams.top_p is None
        assert obj.hyperparams.max_tokens == 20
        assert obj.response_probabilities == pytest.approx(([' Hi', ' Hello'], [0.8, 0.2]))
        assert obj.response_topk[0] == ['Hi', 'Hello']
        assert obj.response_topk[1] == pytest.approx(
            [{' Hi': -0.2, ' Hey': -0.9}, {' Hello': -0.5, ' World': -1.1}],
        )
        assert obj.sampling_method == 'power_mcmc'
        assert obj.branch_sampler == 'metropolis'
        assert obj._written is True

    def test_roundtrip_preserves_all_fields(
        self,
        sample_full_data: LLMOutputDataFull,
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        """A write-then-read cycle preserves all fields exactly."""
        out = tmp_path / 'round.json'
        sample_full_data.write(str(out))

        recovered = self._blank_obj()
        recovered.read(str(out))

        assert recovered.context == sample_full_data.context
        assert recovered.hyperparams.alpha == pytest.approx(sample_full_data.hyperparams.alpha)
        assert recovered.hyperparams.top_k == sample_full_data.hyperparams.top_k
        assert recovered.hyperparams.top_p == sample_full_data.hyperparams.top_p
        assert recovered.hyperparams.max_tokens == sample_full_data.hyperparams.max_tokens
        assert recovered.response_probabilities == pytest.approx(sample_full_data.response_probabilities)
        assert recovered.response_topk[0] == sample_full_data.response_topk[0]
        assert recovered.response_topk[1] == pytest.approx(sample_full_data.response_topk[1])
        assert recovered.sampling_method == sample_full_data.sampling_method