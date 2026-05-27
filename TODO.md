# To-do List

## UI

* Set up a config for the CLI. Location in which to search for .gguf files is easily handled in scripts, but when running the CLI entrypoint it's hard-coded in as `~/.problm-solver/models`.
* Add support for loading `pytorch_model.bin` files, since they're frequently distributed on huggingface. Potentially add in-program support for quantising said models but Llama's own scripts are not too hard to use.
* Investigate porting the API to R.

## Optimisation

* Ensure consistent use of numpy where a speedup could be had, and ensure minimal type conversions.

## Infrastructure

* Add support for optional CUDA use
    * Add support for HPC runs (using slurm)
* `huggingface_hub` etc. should be optional dependencies, and if the user doesn't want them but does want to use MATH500 or other datasets, they will have to point the program towards downloaded `.jsonl` or `.csv` files to import with pandas.
* Make sure installation via `conda` or `mamba` is supported. This is required for various HPC systems.

## Features

* User specification for Llama API beam sampling? When using a custom sampling algorithm, I might have to reimplement beam sampling from first principles.
* An extremely greedy branch sampler could approximate the behaviour of Metropolis sampling, since picking the most probably token in each branch prediction will necessarily minimise the branch logprob. Ask about how I might go about proving this.

## Fixes


## DONE

* Optimise the presently implemented Metropolis method, to allow for longer lookahead depths (more accurate approximation of the power distribution). Should eval the probability of a branch of N tokens from a single Llama call, rather than one per token.
* The result of using a custom sampling algorithm with logits enabled ought to return a richer data dictionary. Investigate this. Maybe the user can specify which kind of data container to use. (light, verbose, probs?)
* Use lower level Llama API (migration in progress), in order to allow for evaluations without updating the context/KV cache.
* Llama can't use KV caching when performing low-temp or power distribution sampling. Could look into saving KV state into RAM before branch sampling and then returning to it after a token or block of tokens has been accepted.
* `SamplePowerDist` doesn't presently include probabilities from past tokens, only future branches
* `BranchSampler` really ought to implement its own logic for calculating future logprobs.
* Make saving and loading KV state silent, include progress tracking (less stdout waffle effectively). Could also look into logging/live-display of current tokens outputted in stdout.
* Investigate use of simultaneous top-p sampling, where one token is overwhelmingly likely to be chosen. Could have the sampler accumulate post-adjust probabilities and terminate upon reaching either top-p or top-k. This would cut down inference time for highly deterministic strings of tokens.
* Add support for top-p sampling techniques. Ideally these should be usable alongside custom sampling algorithms.
* Support for testing on datasets like MATH500 (shouldn't be too hard via script, might be difficult via CLI).
