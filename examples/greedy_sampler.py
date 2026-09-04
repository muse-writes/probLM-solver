"""Naive greedy sampler for probLM-solver."""

import numpy as np

from problm_solver.samplers import AdjustFn, GenerationContext
from problm_solver.candidates import CandidateTokens
from problm_solver.llama_interface import Model

model = Model('~/.problm_solver/Qwen3-4B-Base-Q4_K_M.gguf', 'Why is the sky blue?', logits_all=True)

# Is a Callable[[GenerationContext], CandidateTokens].
def greedy_sampler(ctx: GenerationContext) -> CandidateTokens:
    """My greedy sampler."""
