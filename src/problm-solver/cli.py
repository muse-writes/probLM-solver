"""Command line interface main loop contained here."""

from os.path import splitext
from pathlib import Path
from datetime import datetime

from llama_interface import ModelInstance

PROBLM_DIR = Path.home() / '.problm-solver'
MODELS_DIR = PROBLM_DIR / 'models'
DATA_DIR = PROBLM_DIR / 'datasets'


def ensure_models_dir() -> Path:
    """Create the models directory if it doesn't exist and return its path."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR


def ensure_data_dir() -> Path:
    """Create the data directory if it doesn't exist and returns its path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def list_models() -> list[Path]:
    """Return a sorted list of GGUF files in the models directory."""
    return sorted(MODELS_DIR.glob('*.gguf'))


def get_data_path(model_path) -> Path:
    """Returns a Path variable for data storage.
    """
    timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    return DATA_DIR / (
        splitext(model_path.name)[0] + '_' + timestamp + '.jsonl'
    )


def ui_select_model() -> Path:
    """Display available GGUF models and prompt the user to pick one."""
    models = list_models()

    if not models:
        print(f'No .gguf files found in {MODELS_DIR}')
        print('Place your GGUF model files there and try again.')
        raise SystemExit(1)

    print('Available models:')
    for i, model in enumerate(models, start=1):
        print(f'  [{i}] {model.name}')

    while True:
        choice = input(f'\nSelect a model (1-{len(models)}): ').strip()
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            selected = models[int(choice) - 1]
            print(f'\nUsing: {selected.name}\n')
            return selected
        print('Invalid choice, try again.')


def ui_save_data(fname: str, data) -> None:
    """Handles user interface for saving LLM data.
    """
    resolved = False
    while not resolved:
        response = input(f'Save data to file {fname}? Y/n : ')
        if response is None or response.lower() == 'y':
            ensure_data_dir()
            data.write(fname)
            resolved = True
        elif response.lower() == 'n':
            print('Aborted saving.')
            resolved = True


def main():
    """Main CLI loop."""
    ensure_models_dir()
    model_path = ui_select_model()

    context = input('Enter your prompt: ').strip()
    if not context:
        print('Empty prompt, exiting.')
        raise SystemExit(1)

    model = ModelInstance(str(model_path), context)

# Data generation.
    data_size = int(input('Enter the number of samples to take: ').strip())
    while not isinstance(data_size, int):
        data_size = int(input('Please enter an integer: ').strip())

    data = model.generate_data(data_size)
    print('Data generation complete.')

# Handle saving data.
    data_path = get_data_path(model_path)
    ui_save_data(str(data_path), data)


if __name__ == '__main__':
    main()
