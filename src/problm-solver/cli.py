"""Command line interface main loop contained here."""

from pathlib import Path

from llama_interface import ModelInstance

MODELS_DIR = Path.home() / '.problm-solver' / 'models'


def ensure_models_dir() -> Path:
    """Create the models directory if it doesn't exist and return its path."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR


def list_models() -> list[Path]:
    """Return a sorted list of GGUF files in the models directory."""
    return sorted(MODELS_DIR.glob('*.gguf'))


def select_model() -> Path:
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


def main():
    """Main CLI loop."""
    ensure_models_dir()
    model_path = select_model()

    context = input('Enter your prompt: ').strip()
    if not context:
        print('Empty prompt, exiting.')
        raise SystemExit(1)

    model = ModelInstance(str(model_path), context)
    print(model.query())


if __name__ == '__main__':
    main()
