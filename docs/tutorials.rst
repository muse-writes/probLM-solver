Tutorials
=========

.. highlight:: python
   :linenothreshold: 10

CLI
---

To get started with using the CLI for already-implemented samplers and generating large numbers of outputs, to serve the app straight from the command line, run

.. sourcecode:: bash

   $ docker compose up app

You will be asked to provide a ``.gguf`` model file in the directory ``~/.problm-solver/models``

Scripting with ``problm_solver``
--------------------------------

The main class that probLM-solver uses for interacting with an LLM is ``problm_solver.llama_interface.ModelInstance``.
A simple script for setting up a model, and generating a response with low-temperature sampling, can be seen in the code block below.

.. sourcecode:: python
   :linenos:
   :name: low-temp

   from problm_solver.adjust_probs import SampleLowTemp
   from problm_solver.llama_interface import ModelInstance

   temp = 0.5
   top_k = 8
   model = ModelInstance('path/to/model.gguf', 'Why is the sky blue?', logits_all=True)

   sampling_func = SampleLowTemp(alpha=1./temp)
   data = model.generate_adjusted(
       top_k=top_k,
       top_p=0.9,
       adjust_fn=sampling_func,
       max_tokens=128
   )


Let's explain in detail what this code is doing.

The key idea here is that sampling functions can be defined or freely switched between when generating data, as they aren't specified until ``ModelInstance.generate_adjusted()`` is called.

A couple of hyperparameters are specified in advance, namely the temperature (``temp``), of the model and the number of most-probable tokens considered at each step (``top_k``). If you have already heard of low-temperature sampling or top-k sampling, these concepts will be familiar.

Next, the model stored at a given path is instanced with some context. Importantly, the parameter ``logits_all=True`` is set. If this is left blank, i.e. ``False``, then custom sampling functions will fail when trying to access the model's native token probabilities in order to adjust them.

Then, a sampling function is generated from one of the built-in classes. ``SampleLowTemp()`` is a callable class, with relevant hyperparameters set when it is instanced.

Finally, data from the response are collected into a dataclass (``data``), as an output of ``generate_adjusted()``.
This dictionary is an instance of ``problm_solver.data.LLMOutputDataFull``, and a list of the values contained within is as follows:

.. sourcecode:: python
   :linenos:
   :name: data-dict

   from problm_solver.data import LLMOutputDataFull

   ...

   data.context # Is a tokenised version of the context provided to the model.
   data.hyperparams # A nested dataclass containing the hyperparameters used.
   data.response_probabilities # A tuple of chosen tokens and their (post-adjustment) probability of being chosen.
   data.response_topk # The top-k candidate tokens at each step, and their logarithmic probabilities.
   data.sampling_method # User defined label for the method used.
   data.branch_sampler # User defined label for the branch sampler used. (More on that later.)


Sampling from the Power Distribution
------------------------------------

Recent developments in LLM sampling methods have pointed out that sharper distributions yield more favourable outputs.
Whilst low-temperature sampling goes a way towards solving this issue, it only considers previous tokens outputted when sampling, rather than favouring tokens that may have potential future minima.

At considerable computational expense, sampling from the power distribution can be performed by performing look-ahead runs, and utilising their probabilities before sampling.

This code implements a general framework for sampling from the power distribution.
Let's start with the code snippet below. In this example we will be using the in-built Metropolis method to sample potential future branches, inspired by the implementation from `Karan and Du`_.

.. sourcecode:: python
   :linenos:
   :name: power-dist

   from problm_solver.adjust_probs import MetropolisSampler, SamplePowerDist
   from problm_solver.llama_interface import ModelInstance

   temp = 0.5
   top_k = 8
   peek = 10
   model = ModelInstance('path/to/model.gguf', 'Why is the sky blue?', logits_all=True)

   sampling_func = SamplePowerDist(
       alpha=1./temp,
       lookahead_depth=peek,
       branch_sampler=MetropolisSampler(equil_branches=5, max_branches=10)
   )
   data = model.generate_adjusted(
       top_k=top_k,
       top_p=0.9,
       adjust_fn=sampling_func,
       max_tokens=128
   )

probLM-solver *again* implements the sampling function as a callable class, in ``problm_solver.adjust_probs.SamplePowerDist``.
This takes three main parameters:

- ``alpha``, which you are already familiar with;
- ``lookahead_depth``, which controls how many future tokens the branch sampler can generate when assessing probabilities;
- and ``branch_sampler``, which takes a class which defines the logic for generating and accepting new branches.

We can see that ``MetropolisSampler`` is instanced with two variables.
``equil_branches`` is the number of branches to generate and then discard from probability and convergence calculations, as the method is still locating a minimum. 
``max_branches`` is the maximum number of branches to sample for each of the top-k most probable tokens.


Creating a Branch Sampler
-------------------------

If we take a look at the header for ``MetropolisSampler``:

.. sourcecode:: python
   :name: metropolis-sampler

   class MetropolisSampler(BranchSampler):


we can see that it inherits from an abstract base class called ``BranchSampler``.

New branch sampling algorithms for generating tokens can be defined by the user based on ``BranchSampler``.
It implements three template functions:

.. sourcecode:: python
   :linenos:
   :name: branch-sampler

   class BranchSampler(ABC):

       def reset():
           """Resets stateful samplers."""

       @abstractmethod
       def step(
           self,
           proposed_log_prob: float,
           alpha: float = 1.0,
           forward_log_q: float = 0.0,
           reverse_log_q: float = 0.0,
       ) -> float:
           """Processes the sampler's state and returns the accepted branch."""
           ...

       @abstractmethod
       def should_continue(self, branch_log_probs: npt.NDArray[np.float64]) -> bool:
           """Control flow logic for deciding when the algorithm is finished."""
           ...


- ``BranchSampler.reset()`` ought to reset any state variables (the Metropolis method tracks the current most-favourible logarithmic probability, for instance).
- ``BranchSampler.step()`` controls the criteria for accepting or rejecting a new logarithmic probability and returns the chosen probability.
- ``BranchSampler.should_continue()`` handles logic for terminating a branch sampling algorithm.

All of these methods are called at some point by ``ModelInstance.generate_adjusted()``.

A user who has made the class ``MySampler``, can now pass it to an instance of ``SamplePowerDist``, to create a sampling function.

.. sourcecode:: python
   :linenos:
   :name: power-sampler

   from problm_solver.adjust_probs import BranchSampler, SamplePowerDist

   class MySampler(BranchSampler):
       ...

   temp = 0.5
   sampling_func = SamplePowerDist(alpha=1./temp, lookahead_depth=10, branch_sampler=MySampler())
   data = model.generate_adjusted(top_k=8, top_p=0.9, adjust_fn=sampling_func, max_tokens=128)

Let's explore a hypothetical greedy branch sampler. It might be defined in the following way:

.. sourcecode:: python
   :linenos:
   :name: greedy-sampler

   from problm_solver.adjust_probs import BranchSampler, SamplePowerDist

   class MyGreedySampler(BranchSampler):
       """My greedy sampler, always picks the most probable branch."""

       def __init__(self, max_branches: int = 50):
           """Initialisation method."""
           self._max_branches = max_branches

   # Stateful parameters.
           self._current_log_prob: float | None = None

       def reset(self) -> None:
           """Reset state."""
           self._current_log_prob = None

       def step(self, proposed_log_prob: float) -> float:
           """Calculate and return new best log-prob."""
           if (self._current_log_prob = None
               or proposed_log_prob > self._current_log_prob):
               self._current_log_prob = proposed_log_prob
           return self._current_log_prob

       def should_continue(self, branch_log_probs) -> bool:
           """Control flow for halting algorithm."""
            n = len(branch_log_probs)
            if n >= self._max_branches:
                return False


You can then use this sampler in place of any other.


.. _Karan and Du: https://arxiv.org/abs/2510.14901
