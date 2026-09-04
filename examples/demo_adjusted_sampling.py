"""Demo: initialise a model and run adjust_identity + low-temp sampling."""

from pathlib import Path

from problm_solver.samplers import SampleLowTemp, adjust_identity
from problm_solver.llama_interface import ModelInstance

MODEL = Path.home() / '.problm-solver' / 'models' / 'Qwen3.5-0.8B-Q4_K_M.gguf'
PROMPT = 'Why is the sky blue?'
TOP_K, TOP_P, MAX_TOKENS, ALPHA = 40, 0.9, 128, 2.0

# logits_all=True is required by generate_adjusted() (it raises otherwise).
# n_ctx must hold the formatted prompt + max_tokens.
model = ModelInstance(
    fname=str(MODEL),
    context=PROMPT,
    n_ctx=2048,
    logits_all=True,
)

# No manual reset is needed between calls.
identity = model.generate_adjusted(
    top_k=TOP_K, top_p=TOP_P, adjust_fn=adjust_identity, max_tokens=MAX_TOKENS,
)

low_temp = model.generate_adjusted(
    top_k=TOP_K, top_p=TOP_P,
    adjust_fn=SampleLowTemp(alpha=ALPHA),
    max_tokens=MAX_TOKENS, alpha=ALPHA, sampling_method='LowTemp',
)

for label, data in (('adjust_identity', identity), ('SampleLowTemp', low_temp)):
    tokens, probs = data.response_probabilities
    mean = sum(probs) / len(probs) if probs else 0.0
    print(f'\n=== {label} ===')
    print(f'alpha={data.hyperparams.alpha} top_k={data.hyperparams.top_k} '
          f'top_p={data.hyperparams.top_p} max_tokens={data.hyperparams.max_tokens}')
    print(f'{len(tokens)} tokens, mean prob={mean:.4f}')
    print(''.join(tokens))
