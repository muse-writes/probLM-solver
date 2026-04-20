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


class TestGetDataPath:
    """Tests for get_data_path()."""

    def test_returns_jsonl_extension(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_data_path() always produces a .jsonl path."""
        from problm_solver import cli

        monkeypatch.setattr(cli, 'DATA_DIR', Path('/fake/data'))
        result = cli.get_data_path(Path('/models/my_model.gguf'))
        assert result.suffix == '.jsonl'

    def test_filename_contains_model_stem(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_data_path() includes the model filename stem (without extension)."""
        from problm_solver import cli

        monkeypatch.setattr(cli, 'DATA_DIR', Path('/fake/data'))
        result = cli.get_data_path(Path('/models/my_model.gguf'))
        assert result.name.startswith('my_model_')

    def test_output_is_inside_data_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_data_path() places the output file inside DATA_DIR."""
        from problm_solver import cli

        fake_dir = Path('/fake/data')
        monkeypatch.setattr(cli, 'DATA_DIR', fake_dir)
        result = cli.get_data_path(Path('/models/my_model.gguf'))
        assert result.parent == fake_dir


class TestEnsureDirs:
    """Tests for ensure_models_dir() and ensure_data_dir()."""

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

    def test_ensure_data_dir_creates_directory(
        self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ensure_data_dir() creates the directory if it does not exist."""
        from problm_solver import cli

        target = tmp_path / 'datasets'
        monkeypatch.setattr(cli, 'DATA_DIR', target)
        assert not target.exists()
        cli.ensure_data_dir()
        assert target.is_dir()

    def test_ensure_data_dir_returns_path(
        self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ensure_data_dir() returns the DATA_DIR path."""
        from problm_solver import cli

        target = tmp_path / 'datasets'
        monkeypatch.setattr(cli, 'DATA_DIR', target)
        result = cli.ensure_data_dir()
        assert result == target


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

        monkeypatch.setattr(cli, 'ensure_data_dir', lambda: tmp_path)
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
