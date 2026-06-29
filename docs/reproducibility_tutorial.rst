Reproducibility and Random Seeds
================================

This tutorial explains how to make probabilistic generation reproducible in
``probLM-solver``.

Why this matters
----------------

Sampling-based decoding is stochastic. If you do not control randomness,
outputs may vary between runs even with the same prompt, model, and
hyperparameters.

The package provides a ``RandomManager`` to coordinate deterministic random
streams from one root seed.

Minimal reproducible script
---------------------------

.. sourcecode:: python
   :linenos:

   from pathlib import Path

   from problm_solver.adjust_probs import SampleLowTemp, adjust_identity
   from problm_solver.llama_interface import ModelInstance
   from problm_solver.random import RandomManager

   model_path = Path.home() / '.problm-solver' / 'models' / 'Qwen3.5-0.8B-Q4_K_M.gguf'
   prompt = 'Why is the sky blue?'

   # One seed for the whole run.
   run_rng = RandomManager(seed=12345)

   model = ModelInstance(
       fname=str(model_path),
       context=prompt,
       logits_all=True,
       rng=run_rng,
   )

   out_identity = model.generate_adjusted(
       top_k=30,
       top_p=0.9,
       adjust_fn=adjust_identity,
       max_tokens=128,
       sampling_method='identity',
   )

   model.change_context(prompt)

   out_low_temp = model.generate_adjusted(
       top_k=30,
       top_p=0.9,
       adjust_fn=SampleLowTemp(alpha=2.0),
       max_tokens=128,
       sampling_method='low_temp_alpha2',
   )

   print('identity:', ''.join(out_identity.response_probabilities[0]))
   print('low-temp:', ''.join(out_low_temp.response_probabilities[0]))

Running this script again with the same seed reproduces the same outputs
(assuming same model file, package version, and runtime environment).

RNG inputs accepted by user-facing APIs
---------------------------------------

Methods that consume randomness accept ``rng`` as an ``RNGLike`` value:

- ``numpy.random.Generator``: use your own generator directly.
- ``int``: interpreted as a seed.
- ``RandomManager``: use named deterministic streams from one root seed.
- ``None``: use default non-deterministic behavior.

Example with explicit per-call seed:

.. sourcecode:: python

   text = model.query(rng=2026)

Example with user-managed generator:

.. sourcecode:: python

   import numpy as np

   gen = np.random.default_rng(2026)
   text_a = model.query(rng=gen)
   text_b = model.query(rng=gen)

Notes
-----

For strict reproducibility across machines, pin:

- model file/version,
- ``problm_solver`` version,
- Python/numpy versions,
- and any hardware-specific inference settings.
