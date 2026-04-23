"""Tests for CLI utility functions in cli.py."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from problm_solver.data import LLMOutputData


class TestListModels:
    """Tests for list_models()."""

    def test_returns_empty_list_when_no_gguf_files(
        self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """list_models() returns [] when the models directory has no .gguf files."""
        from problm_solver import cli

        monkeypatch.setattr(cli, 'MODELS_DIR', tmp_path)
        assert cli.list_models() == []

    def test_excludes_non_gguf_files(
        self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """list_models() ignores files that don't end in .gguf."""
        from problm_solver import cli

        monkeypatch.setattr(cli, 'MODELS_DIR', tmp_path)
        (tmp_path / 'model.bin').touch()
        (tmp_path / 'readme.txt').touch()
        assert cli.list_models() == []

    def test_returns_only_gguf_files(
        self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """list_models() returns only .gguf files."""
        from problm_solver import cli

        monkeypatch.setattr(cli, 'MODELS_DIR', tmp_path)
        gguf = tmp_path / 'model.gguf'
        gguf.touch()
        (tmp_path / 'other.bin').touch()
        result = cli.list_models()
        assert result == [gguf]

    def test_results_are_sorted(
        self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """list_models() returns files in alphabetical order."""
        from problm_solver import cli

        monkeypatch.setattr(cli, 'MODELS_DIR', tmp_path)
        for name in ['zeta.gguf', 'alpha.gguf', 'beta.gguf']:
            (tmp_path / name).touch()
        names = [p.name for p in cli.list_models()]
        assert names == sorted(names)


class TestGetResponsesPath:
    """Tests for get_responses_path()."""

    def test_returns_jsonl_extension(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_responses_path() always produces a .jsonl path."""
        from problm_solver import cli

        monkeypatch.setattr(cli, 'RESPONSES_DIR', Path('/fake/responses'))
        result = cli.get_responses_path(Path('/models/my_model.gguf'))
        assert result.suffix == '.jsonl'

    def test_filename_contains_model_stem(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_responses_path() includes the model filename stem."""
        from problm_solver import cli

        monkeypatch.setattr(cli, 'RESPONSES_DIR', Path('/fake/responses'))
        result = cli.get_responses_path(Path('/models/my_model.gguf'))
        assert result.name.startswith('my_model_')

    def test_output_is_inside_responses_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_responses_path() places the output file inside RESPONSES_DIR."""
        from problm_solver import cli

        fake_dir = Path('/fake/responses')
        monkeypatch.setattr(cli, 'RESPONSES_DIR', fake_dir)
        result = cli.get_responses_path(Path('/models/my_model.gguf'))
        assert result.parent == fake_dir


class TestGetProbsPath:
    """Tests for get_probs_path()."""

    def test_returns_json_extension(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_probs_path() always produces a .json path."""
        from problm_solver import cli

        monkeypatch.setattr(cli, 'PROBS_DIR', Path('/fake/probs'))
        result = cli.get_probs_path(Path('/models/my_model.gguf'))
        assert result.suffix == '.json'

    def test_filename_contains_model_stem(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_probs_path() includes the model filename stem."""
        from problm_solver import cli

        monkeypatch.setattr(cli, 'PROBS_DIR', Path('/fake/probs'))
        result = cli.get_probs_path(Path('/models/my_model.gguf'))
        assert 'my_model' in result.name

    def test_filename_has_prob_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_probs_path() prefixes the filename with 'prob_'."""
        from problm_solver import cli

        monkeypatch.setattr(cli, 'PROBS_DIR', Path('/fake/probs'))
        result = cli.get_probs_path(Path('/models/my_model.gguf'))
        assert result.name.startswith('prob_')

    def test_output_is_inside_probs_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_probs_path() places the output file inside PROBS_DIR."""
        from problm_solver import cli

        fake_dir = Path('/fake/probs')
        monkeypatch.setattr(cli, 'PROBS_DIR', fake_dir)
        result = cli.get_probs_path(Path('/models/my_model.gguf'))
        assert result.parent == fake_dir


class TestEnsureDirs:
    """Tests for ensure_models_dir(), ensure_responses_dir(), and ensure_probs_dir()."""

    def test_ensure_models_dir_creates_directory(
        self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ensure_models_dir() creates the directory if it does not exist."""
        from problm_solver import cli

        target = tmp_path / 'models'
        monkeypatch.setattr(cli, 'MODELS_DIR', target)
        assert not target.exists()
        cli.ensure_models_dir()
        assert target.is_dir()

    def test_ensure_models_dir_returns_path(
        self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ensure_models_dir() returns the MODELS_DIR path."""
        from problm_solver import cli

        target = tmp_path / 'models'
        monkeypatch.setattr(cli, 'MODELS_DIR', target)
        result = cli.ensure_models_dir()
        assert result == target

    def test_ensure_responses_dir_creates_directory(
        self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ensure_responses_dir() creates the directory if it does not exist."""
        from problm_solver import cli

        target = tmp_path / 'datasets' / 'responses'
        monkeypatch.setattr(cli, 'RESPONSES_DIR', target)
        assert not target.exists()
        cli.ensure_responses_dir()
        assert target.is_dir()

    def test_ensure_responses_dir_returns_path(
        self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ensure_responses_dir() returns the RESPONSES_DIR path."""
        from problm_solver import cli

        target = tmp_path / 'datasets' / 'responses'
        monkeypatch.setattr(cli, 'RESPONSES_DIR', target)
        result = cli.ensure_responses_dir()
        assert result == target

    def test_ensure_probs_dir_creates_directory(
        self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ensure_probs_dir() creates the directory if it does not exist."""
        from problm_solver import cli

        target = tmp_path / 'datasets' / 'probabilities'
        monkeypatch.setattr(cli, 'PROBS_DIR', target)
        assert not target.exists()
        cli.ensure_probs_dir()
        assert target.is_dir()

    def test_ensure_probs_dir_returns_path(
        self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ensure_probs_dir() returns the PROBS_DIR path."""
        from problm_solver import cli

        target = tmp_path / 'datasets' / 'probabilities'
        monkeypatch.setattr(cli, 'PROBS_DIR', target)
        result = cli.ensure_probs_dir()
        assert result == target


class TestUiSelectFunction:
    """Tests for ui_select_function()."""

    def test_returns_1_for_gen_data_choice(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ui_select_function() returns 1 when the user selects option 1."""
        from problm_solver import cli

        monkeypatch.setattr('builtins.input', lambda _: '1')
        assert cli.ui_select_function() == 1

    def test_returns_2_for_probs_choice(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ui_select_function() returns 2 when the user selects option 2."""
        from problm_solver import cli

        monkeypatch.setattr('builtins.input', lambda _: '2')
        assert cli.ui_select_function() == 2

    def test_reprompts_on_invalid_input(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ui_select_function() keeps prompting until a valid choice is entered."""
        from problm_solver import cli

        responses = iter(['0', '-1', 'abc', '2'])
        monkeypatch.setattr('builtins.input', lambda _: next(responses))
        assert cli.ui_select_function() == 2


class TestUiSelectModel:
    """Tests for ui_select_model()."""

    def test_raises_system_exit_when_no_models(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ui_select_model() calls SystemExit(1) when the models directory is empty."""
        from problm_solver import cli

        monkeypatch.setattr(cli, 'list_models', lambda: [])
        with pytest.raises(SystemExit) as exc_info:
            cli.ui_select_model()
        assert exc_info.value.code == 1

    def test_returns_selected_model_path(
        self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ui_select_model() returns the Path of the model chosen by the user."""
        from problm_solver import cli

        models = [tmp_path / 'alpha.gguf', tmp_path / 'beta.gguf']
        monkeypatch.setattr(cli, 'list_models', lambda: models)
        monkeypatch.setattr('builtins.input', lambda _: '2')
        result = cli.ui_select_model()
        assert result == models[1]

    def test_reprompts_on_invalid_input(
        self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ui_select_model() keeps prompting until the user enters a valid choice."""
        from problm_solver import cli

        models = [tmp_path / 'only.gguf']
        responses = iter(['0', 'abc', '99', '1'])  # first three are invalid
        monkeypatch.setattr(cli, 'list_models', lambda: models)
        monkeypatch.setattr('builtins.input', lambda _: next(responses))
        result = cli.ui_select_model()
        assert result == models[0]


class TestUiSaveData:
    """Tests for ui_save_data()."""

    def _make_mock_data(self) -> MagicMock:
        """Return a MagicMock that stands in for LLMOutputData."""
        mock = MagicMock(spec=LLMOutputData)
        mock.data = np.array(['response'], dtype=str)
        return mock

    def test_calls_write_when_user_confirms(
        self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ui_save_data() calls data.write() when the user enters 'y'."""
        from problm_solver import cli

        monkeypatch.setattr(cli, 'ensure_responses_dir', lambda: tmp_path)
        monkeypatch.setattr('builtins.input', lambda _: 'y')
        mock_data = self._make_mock_data()
        cli.ui_save_data('out.jsonl', mock_data)
        mock_data.write.assert_called_once_with('out.jsonl')

    def test_does_not_call_write_when_user_declines(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ui_save_data() does not call data.write() when the user enters 'n'."""
        from problm_solver import cli

        monkeypatch.setattr('builtins.input', lambda _: 'n')
        mock_data = self._make_mock_data()
        cli.ui_save_data('out.jsonl', mock_data)
        mock_data.write.assert_not_called()


class TestUiSaveTokenData:
    """Tests for ui_save_token_data()."""

    def _make_mock_token_data(self) -> MagicMock:
        """Return a MagicMock standing in for LLMTokenData."""
        from problm_solver.data import LLMTokenData

        return MagicMock(spec=LLMTokenData)

    def test_calls_write_when_user_confirms(
        self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ui_save_token_data() calls data.write() when the user enters 'y'."""
        from problm_solver import cli

        monkeypatch.setattr(cli, 'ensure_probs_dir', lambda: tmp_path)
        monkeypatch.setattr('builtins.input', lambda _: 'y')
        mock_data = self._make_mock_token_data()
        cli.ui_save_token_data('out.json', mock_data)
        mock_data.write.assert_called_once_with('out.json')

    def test_does_not_call_write_when_user_declines(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ui_save_token_data() does not call data.write() when the user enters 'n'."""
        from problm_solver import cli

        monkeypatch.setattr('builtins.input', lambda _: 'n')
        mock_data = self._make_mock_token_data()
        cli.ui_save_token_data('out.json', mock_data)
        mock_data.write.assert_not_called()


class TestUiGetProbs:
    """Tests for ui_get_probs()."""

    def test_calls_query_log_probs(
        self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ui_get_probs() calls model.query_log_probs() exactly once."""
        from problm_solver import cli

        mock_model = MagicMock()
        mock_model.query_log_probs.return_value = MagicMock()
        monkeypatch.setattr(cli, 'get_probs_path', lambda _: tmp_path / 'out.json')
        monkeypatch.setattr(cli, 'ui_save_token_data', lambda *_: None)
        cli.ui_get_probs(mock_model, tmp_path / 'model.gguf')
        mock_model.query_log_probs.assert_called_once()

    def test_passes_result_to_ui_save_token_data(
        self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ui_get_probs() passes the query_log_probs() result to ui_save_token_data."""
        from problm_solver import cli

        token_data = MagicMock()
        mock_model = MagicMock()
        mock_model.query_log_probs.return_value = token_data
        saved = {}
        monkeypatch.setattr(cli, 'get_probs_path', lambda _: tmp_path / 'out.json')
        monkeypatch.setattr(cli, 'ui_save_token_data', lambda fname, data: saved.update({'fname': fname, 'data': data}))
        cli.ui_get_probs(mock_model, tmp_path / 'model.gguf')
        assert saved['data'] is token_data

    def test_passes_probs_path_to_ui_save_token_data(
        self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ui_get_probs() passes the path from get_probs_path() to ui_save_token_data."""
        from problm_solver import cli

        expected_path = tmp_path / 'prob_model_2026.json'
        mock_model = MagicMock()
        mock_model.query_log_probs.return_value = MagicMock()
        saved = {}
        monkeypatch.setattr(cli, 'get_probs_path', lambda _: expected_path)
        monkeypatch.setattr(cli, 'ui_save_token_data', lambda fname, data: saved.update({'fname': fname}))
        cli.ui_get_probs(mock_model, tmp_path / 'model.gguf')
        assert saved['fname'] == str(expected_path)
