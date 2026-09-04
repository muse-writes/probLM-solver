:orphan:

A Dead Simple Greedy Sampler
============================

Overview
--------

This tutorial builds the smallest possible custom sampler from scratch: a
**greedy** sampler that always emits the single most-probable next token.

It is intentionally short — short enough to type out live and talk through in
a presentation. By the end you will have written your own
:data:`AdjustFn` and run it through
:meth:`Model.generate_with_sampler()`.

Prerequisites
-------------

- A working ``problm_solver`` install.
- A local ``.gguf`` model under ``~/.problm-solver/models``.

The whole idea
--------------

An :data:`AdjustFn` is just a callable. It receives a
:class:`SamplerContext` holding the current top-k candidate tokens and
their log-probabilities, and returns a :class:`CandidateTokens` whose
log-probabilities will be renormalised and sampled from.

Greedy sampling means: collapse the distribution onto its argmax. Give the
most-probable token probability ``1`` (log-prob ``0``) and every other
candidate probability ``0`` (log-prob ``-inf``).

The sampler
-----------

.. sourcecode:: python
   :linenos:
   :name: greedy-fn

   import numpy as np
   from problm_solver.samplers import CandidateTokens, SamplerContext

   def greedy(context: SamplerContext) -> CandidateTokens:
       """Always pick the single most-probable candidate token."""
       ids = context.token_id_probs.candidate_ids
       lps = context.token_id_probs.candidate_logprobs

       best = int(np.argmax(lps))
       new_lps = np.full_like(lps, -np.inf)
       new_lps[best] = 0.0
       return CandidateTokens(candidate_ids=ids, candidate_logprobs=new_lps)

That is the entire sampler — three lines of logic.

Using it
--------

Pass ``greedy`` as the ``adjust_fn`` exactly like any built-in sampler:

.. sourcecode:: python
   :linenos:
   :name: greedy-use

   from problm_solver.llama_interface import Model

   model = Model('path/to/model.gguf', 'Why is the sky blue?', logits_all=True)

   data = model.generate_with_sampler(
       top_k=40,
       top_p=1.0,
       adjust_fn=greedy,
       max_tokens=128,
   )

``top_k`` and ``top_p`` decide which candidates reach the adjust function;
``greedy`` then picks the best of them. With ``top_p=1.0`` nothing is trimmed
by probability mass, so all ``top_k`` candidates are in play before the
collapse.

What to say while presenting it
-------------------------------

- An :data:`AdjustFn` takes a :class:`SamplerContext`, returns
  :class:`CandidateTokens`. That is the whole contract.
- ``context.token_id_probs`` already holds the filtered top-k candidates, so
  the sampler never touches the raw vocabulary.
- ``np.argmax`` finds the best token; ``-inf`` everywhere else makes its
  probability exactly 1 after renormalisation. Deterministic by construction.
- ``logits_all=True`` on the model is required — without it
  ``token_id_probs`` is empty and every custom sampler fails.
- Greedy is just :class:`SampleLowTemp` in the limit ``alpha -> inf``; this
  version makes the limit explicit.

Next steps
----------

- Swap the ``-inf`` collapse for a temperature scaling to recover
  :class:`SampleLowTemp`.
- Replace ``argmax`` with a :class:`BranchSampler` and :class:`SamplePowerDist`
  to look ahead before choosing — see the *Sampling from the Power
  Distribution* section of the main tutorials page.
