# To-do List

## UI

* Add support for loading `pytorch_model.bin` files, since they're frequently distributed on huggingface. Potentially add in-program support for quantising said models but Llama's own scripts are not too hard to use.

## Optimisation

* Optimise the presently implemented Metropolis method, to allow for longer lookahead depths (more accurate approximation of the power distribution). Should eval the probability of a branch of N tokens from a single Llama call, rather than one per token.
