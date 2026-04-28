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

A simple script for setting up and running ``problm_solver`` with low-temperature sampling can be seen in :numref:`low-temp`.

.. sourcecode:: python
   :linenos:
   :name: low-temp

   from problm_solver.adjust_probs import SampleLowTemp
   from problm_solver.llama_interface import ModelInstance

   temp = 0.5
   top_m = 8
   sampling_func = SampleLowTemp(alpha=1./temp)
   model = ModelInstance('path/to/model.gguf', 'Why is the sky blue?', logits_all=True)
   data = model.generate_adjusted(n_tokens=top_m, adjust_fn=sampling_func, max_tokens=128)


Here ``data`` is currently just a response in a dictionary, packaged alongside its prompt and an identifier.


Creating a Branch Sampler
-------------------------

New branch sampling algorithms for generating tokens can be defined by the user based on the ``BranchSampler`` abstract base class.
It implements three template functions:

.. sourcecode:: python
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

Details on how to create one of these exactly are found below.

A user who has made the class ``MySampler``, can now pass it to an instance of ``SamplePowerDist``, to create a sampling function.

.. sourcecode:: python
   :name: power-sampler

   from problm_solver.adjust_probs import SamplePowerDist, BranchSampler

   class MySampler(BranchSampler):
       ...

   temp = 0.5
   sampling_func = SamplePowerDist(alpha=1./temp, lookahead_depth=10, branch_sampler=MySampler())
   data = model.generate_adjusted(n_tokens=8, adjust_fn=sampling_func, max_tokens=128)



