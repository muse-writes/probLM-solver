"""Command line interface main loop contained here."""

import logging
from datetime import UTC, datetime
from os.path import splitext
from pathlib import Path

import pandas as pd

from problm_solver.adjust_probs import (
    MetropolisSampler,
    SampleLowTemp,
    SamplePowerDist,
    adjust_identity,
)
from problm_solver.data import LLMOutputData, LLMOutputDataFull, LLMTokenData
from problm_solver.datasets import get_math500, get_problems_math500
from problm_solver.llama_interface import ModelInstance
from problm_solver.utils import TqdmHandler

PROBLM_DIR = Path.home() / '.problm-solver'
MODELS_DIR = PROBLM_DIR / 'models'
RESPONSES_DIR = PROBLM_DIR / 'datasets' / 'responses'
PROBS_DIR = PROBLM_DIR / 'datasets' / 'probabilities'

NUMBER_OF_FUNCTIONS = 5
GEN_DATA = 1
PROBS = 2
LOW_TEMP = 3
POWER_SAMPLING = 4
MATH500 = 5

logging.basicConfig(
    handlers=[TqdmHandler()],
    level=logging.DEBUG,
    format='%(levelname)s - %(name)s - %(message)s',
)


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


def get_math500_results_path(model_path: Path) -> Path:
    """Return a timestamped .jsonl path inside RESPONSES_DIR for a MATH500 results dataset."""
    timestamp = datetime.now(tz=UTC).strftime('%Y-%m-%d-%H:%M:%S')
    return RESPONSES_DIR / ('math500_' + splitext(model_path.name)[0] + '_' + timestamp + '.jsonl')


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


def ui_save_data(fname: str, data: LLMOutputData | LLMOutputDataFull) -> None:
    """Handle user interface for saving LLM response data."""
    resolved = False
    while not resolved:
        response = input(f'Save data to file {fname}? Y/n : ')
        if response is None or response.lower() == 'y':
            ensure_responses_dir()
            data.write(fname)
            resolved = True
        elif response.lower() == 'n':
            fname = input('Enter alternative file name or leave blank to discard: ')
            if fname is not None:
                fname = str(RESPONSES_DIR / fname)
                data.write(fname)
            else:
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
    print('  [3] Generate a response with low-temp sampling (low_temp run)')
    print('  [4] Generate a response with MCMC power sampling (power_mcmc run)')
    print('  [5] Discard the context and generate answers to MATH500 (math500 run)')
    while True:
        choice = input(f'\nSelect a function (1-{NUMBER_OF_FUNCTIONS}): ').strip()
        if choice.isdigit() and 1 <= int(choice) <= NUMBER_OF_FUNCTIONS:
            return int(choice)
        print('Invalid choice, try again.')


def ui_generate_low_temp(model: ModelInstance, model_path: Path) -> None:
    """Handle user interface for getting model response using low temp sampling."""
    print('\nGenerating output from low-temp sampling.')
    alpha = float(input('Please input the value of alpha, as a float: '))
    sampling_fn = SampleLowTemp(alpha=alpha)
    top_k = int(input(
        'Please input the number of most probable token candidates (M) to consider at each step: '
    ))
    max_tokens = int(input('Please input the maximum number of response tokens: '))
    data = model.generate_adjusted(
        top_k=top_k,
        top_p=0.9,
        adjust_fn=sampling_fn,
        max_tokens=max_tokens,
        alpha=alpha
    )

# Handle saving data
    response_path = get_adjusted_path(model_path)
    ui_save_data(str(response_path), data)


def ui_generate_power_mcmc(model: ModelInstance, model_path: Path) -> None:
    """Handle user interface for getting model response using power sampling with MCMC."""
    print('\nGenerating output using power sampling.')

# Get user input
    alpha = float(input('Please input the value of alpha, as a float: '))
    peek = int(input(
        'Please input the max lookahead depth to generate branches over, '
        'as an integer: '
    ))
    top_k = int(input(
        'Please input the number of most probable token candidates (K) to consider at each step: '
    ))
    max_tokens = int(input('Please input the maximum number of response tokens: '))

# Construct sampling function and generate with it.
    sampling_fn = SamplePowerDist(
        alpha=alpha,
        lookahead_depth=peek,
        branch_sampler=MetropolisSampler(max_branches=10)
    )
    data = model.generate_adjusted(
        top_k=top_k,
        top_p=0.9,
        adjust_fn=sampling_fn,
        max_tokens=max_tokens,
        alpha=alpha,
        sampling_method='Power Distribution',
        branch_sampler='Metropolis Sampler',
    )

# Handle saving data.
    response_path = get_adjusted_path(model_path)
    ui_save_data(str(response_path), data)


def ui_math500(model: ModelInstance, model_path: Path) -> None:
    """Answer MATH500 problems and write response to file."""
    math500_data = get_math500()
    math500_problems = get_problems_math500()

    print('For MATH500 runs, temporarily using hard-coded parameters.')

    sampling_fn = SamplePowerDist(
        alpha=2.0,
        lookahead_depth=20,
        branch_sampler=MetropolisSampler(max_branches=10)
    )

    results = model.test_dataset_adjusted(
        dataset=math500_problems,
        top_k=30,
        top_p=0.9,
        adjust_fn=sampling_fn,
        max_tokens=512
    )

    control = model.test_dataset_adjusted(
        dataset=math500_problems,
        top_k=30,
        top_p=0.9,
        adjust_fn=adjust_identity,
        max_tokens=512
    )

# Combine into a DataFrame alongside the original dataset columns.
    df = math500_data.copy()
    df['power_dist_answer'] = results
    df['control_answer'] = control

# Prompt and save.
    results_path = get_math500_results_path(model_path)
    ensure_responses_dir()
    response = input(f'Save results to {results_path}? Y/n : ').strip()
    if response.lower() != 'n':
        df.to_json(results_path, orient='records', lines=True)
        print(f'Results saved to {results_path}.')
    else:
        print('Results discarded.')


def main() -> None:
    """Perform main CLI."""
    ensure_models_dir()
    model_path = ui_select_model()

    context = input('Enter your prompt: ').strip()
    if not context:
        print('Empty prompt, exiting.')
        raise SystemExit(1)

    choice = ui_select_function()
    use_logits: bool = choice in (PROBS, LOW_TEMP, POWER_SAMPLING, MATH500)
    model = ModelInstance(str(model_path), context, logits_all=use_logits)

    match choice:
        case 1:
            ui_gen_data(model, model_path)
        case 2:
            ui_get_probs(model, model_path)
        case 3:
            ui_generate_low_temp(model, model_path)
        case 4:
            ui_generate_power_mcmc(model, model_path)
        case 5:
            ui_math500(model, model_path)
        case _:
            raise UnexpectedFunctionError


if __name__ == '__main__':
    main()
