"""Naive greedy sampler for probLM-solver."""

import numpy as np

from problm_solver.candidates import CandidateTokens
from problm_solver.llama_interface import Model
from problm_solver.samplers import AdjustFn, SamplerContext

model = Model('~/.problm_solver/Qwen3-4B-Base-Q4_K_M.gguf', 'Why is the sky blue?', logits_all=True)

# Is a Callable[[SamplerContext], CandidateTokens].
def greedy_sampler(ctx: SamplerContext) -> CandidateTokens:
    """My greedy sampler."""
