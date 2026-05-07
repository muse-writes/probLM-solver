# To-do List

## UI

* Add support for loading `pytorch_model.bin` files, since they're frequently distributed on huggingface. Potentially add in-program support for quantising said models but Llama's own scripts are not too hard to use.
* User specification for Llama API beam sampling? When using a custom sampling algorithm, I might have to reimplement beam sampling from first principles.

## Optimisation

* Use lower level Llama API (migration in progress), in order to allow for evaluations without updating the context/KV cache.

## Infrastructure

* Add support for optional CUDA use
    * Add support for HPC runs (using slurm)

## Features

* Llama can't use KV caching when performing low-temp or power distribution sampling. Could look into saving KV state into RAM before branch sampling and then returning to it after a token or block of tokens has been accepted.

## DONE

* Optimise the presently implemented Metropolis method, to allow for longer lookahead depths (more accurate approximation of the power distribution). Should eval the probability of a branch of N tokens from a single Llama call, rather than one per token.
* The result of using a custom sampling algorithm with logits enabled ought to return a richer data dictionary. Investigate this. Maybe the user can specify which kind of data container to use. (light, verbose, probs?)
