"""Command line interface main loop contained here."""

from datetime import UTC, datetime
from os.path import splitext
from pathlib import Path

from problm_solver.adjust_probs import SampleLowTemp
from problm_solver.data import LLMOutputData, LLMTokenData
from problm_solver.llama_interface import ModelInstance

PROBLM_DIR = Path.home() / '.problm-solver'
MODELS_DIR = PROBLM_DIR / 'models'
RESPONSES_DIR = PROBLM_DIR / 'datasets' / 'responses'
PROBS_DIR = PROBLM_DIR / 'datasets' / 'probabilities'

NUMBER_OF_FUNCTIONS = 3
GEN_DATA = 1
PROBS = 2
GENERATE_ADJUSTED = 3


class UnexpectedFunctionError(Exception):
    """Raised when the developer hasn't extended the interface properly."""

    def __init__(self) -> None:
        """Initialize error message."""
        super().__init__('Program function not accounted for in match-case statement')


def ensure_models_dir() -> Path:
    """Create the models directory if it doesn't exist and return its path."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR


def ensure_responses_dir() -> Path:
    """Create the responses directory if it doesn't exist and return its path."""
    RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
    return RESPONSES_DIR


def ensure_probs_dir() -> Path:
    """Create the probabilities directory if it doesn't exist and return its path."""
    PROBS_DIR.mkdir(parents=True, exist_ok=True)
    return PROBS_DIR


def list_models() -> list[Path]:
    """Return a sorted list of GGUF files in the models directory."""
    return sorted(MODELS_DIR.glob('*.gguf'))


def get_responses_path(model_path: Path) -> Path:
    """Return a timestamped .jsonl path inside RESPONSES_DIR for a response dataset."""
    timestamp = datetime.now(tz=UTC).strftime('%Y-%m-%d-%H:%M:%S')
    return RESPONSES_DIR / (splitext(model_path.name)[0] + '_' + timestamp + '.jsonl')

def get_adjusted_path(model_path: Path) -> Path:
    """Return a timestamped .json path inside RESPONSES_DIR for an adjusted response."""
    timestamp = datetime.now(tz=UTC).strftime('%Y-%m-%d-%H:%M:%S')
    return RESPONSES_DIR / (
        'adjusted' + '_' + splitext(model_path.name)[0]
        + '_' + timestamp + '.jsonl'
    )


def get_probs_path(model_path: Path) -> Path:
    """Return a timestamped .json path inside PROBS_DIR for a probability dataset."""
    timestamp = datetime.now(tz=UTC).strftime('%Y-%m-%d-%H:%M:%S')
    return PROBS_DIR / ('prob_' + splitext(model_path.name)[0] + '_' + timestamp + '.json')


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


def ui_gen_data(model: ModelInstance, model_path: Path) -> None:
    """Handle user interface for generating LLM data."""
    data_size = int(input('Enter the number of samples to take: ').strip())
    while not isinstance(data_size, int):
        data_size = int(input('Please enter an integer: ').strip())

    data = model.generate_data(data_size)
    print('Data generation complete.')

    data_path = get_responses_path(model_path)
    ui_save_data(str(data_path), data)


def ui_save_data(fname: str, data: LLMOutputData) -> None:
    """Handle user interface for saving LLM response data."""
    resolved = False
    while not resolved:
        response = input(f'Save data to file {fname}? Y/n : ')
        if response is None or response.lower() == 'y':
            ensure_responses_dir()
            data.write(fname)
            resolved = True
        elif response.lower() == 'n':
            print('Aborted saving.')
            resolved = True


def ui_save_token_data(fname: str, data: LLMTokenData) -> None:
    """Handle user interface for saving LLM token probability data."""
    resolved = False
    while not resolved:
        response = input(f'Save data to file {fname}? Y/n : ')
        if response is None or response.lower() == 'y':
            ensure_probs_dir()
            data.write(fname)
            resolved = True
        elif response.lower() == 'n':
            print('Aborted saving.')
            resolved = True


def ui_get_probs(model: ModelInstance, model_path: Path) -> None:
    """Handle user interface for querying once with logprobs enabled."""
    data = model.query_log_probs()
    print('Token probability query complete.')
    data_path = get_probs_path(model_path)
    ui_save_token_data(str(data_path), data)


def ui_select_function() -> int:
    """Prompt the user to select a program function and return their integer choice."""
    print('\nChoose program function:')
    print('  [1] Generate LLM output data (gen_data run)')
    print('  [2] Get tokens and probabilities (probs run)')
    print('  [3] Generate a response with adjusted probabilities (adjust run)')
    while True:
        choice = input(f'\nSelect a function (1-{NUMBER_OF_FUNCTIONS}): ').strip()
        if choice.isdigit() and 1 <= int(choice) <= NUMBER_OF_FUNCTIONS:
            return int(choice)
        print('Invalid choice, try again.')


def ui_generate_adjusted(model: ModelInstance, model_path: Path) -> None:
    """Handle user interface for getting model response using adjusted probability function."""
    alpha = float(input('Please input the value of alpha, as a float: '))
    sampling_fn = SampleLowTemp(alpha=alpha)
    top_m = int(input(
        'Please input the number of most probable token candidates (M) to consider at each step: '
    ))
    max_tokens = int(input('Please input the maximum number of response tokens: '))
    data = model.generate_adjusted(n_tokens=top_m, adjust_fn=sampling_fn, max_tokens=max_tokens)

# Handle saving data
    response_path = get_adjusted_path(model_path)
    ui_save_data(str(response_path), data)


def main() -> None:
    """Perform main CLI."""
    ensure_models_dir()
    model_path = ui_select_model()

    context = input('Enter your prompt: ').strip()
    if not context:
        print('Empty prompt, exiting.')
        raise SystemExit(1)

    choice = ui_select_function()
    use_logits: bool = choice in (PROBS, GENERATE_ADJUSTED)
    model = ModelInstance(str(model_path), context, logits_all=use_logits)

    match choice:
        case 1:
            ui_gen_data(model, model_path)
        case 2:
            ui_get_probs(model, model_path)
        case 3:
            ui_generate_adjusted(model, model_path)
        case _:
            raise UnexpectedFunctionError


if __name__ == '__main__':
    main()
