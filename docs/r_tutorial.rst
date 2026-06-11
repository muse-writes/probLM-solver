R Tutorial
==========

Overview
--------

This is a quick setup and scripting guide for running ``problm_solver`` in R.

Prerequisites
-------------

- R installed locally.
- The R module ``reticulate`` must be installed to call ``problm_solver``'s API from R.
- A local GGUF model accessible to ``problm_solver``.

Environment setup
-----------------

First, the Python environment must be set up, using either Conda or uv/pip.
See the ``README.md`` file for environment installation instructions

You will not need to be in the environment in order to run R scripts, as that can be set up either in Rstudio or your text editor of choice.

``reticulate`` can be installed using the standard R command ``install.packages("reticulate", repos = "https://cloud.r-project.org")``, but a clean way of ensuring this is always installed and imported in a given R script is to include,

.. sourcecode:: r

   if (!requireNamespace("reticulate", quietly = True)) {
     install.packages("reticulate", repos = "https://cloud.r-project.org")
   }
   library(reticulate)

At the start of a script.

Set up ``problm_solver``
------------------------

Once ``reticulate`` is installed, R must be pointed towards your Python environment. You can do this with one of the following commands

.. sourcecode:: r

   use_condaenv("problm-solver", required = TRUE)
   py_config()

If Conda is used. Alternatively, any Python virtual environment can be used with

.. sourcecode:: r

   use_python("path/to/venv/bin/python", required = TRUE)
   py_config()

replacing the relevant path with the path to your Python binary.

Finally, the modules themselves can be imported and assigned to R variables. For many standard sampling tasks, these imports should suffice:

.. sourcecode:: r

   llama <- import("problm_solver.llama_interface")
   adj <- import("problm_solver.adjust_probs")

Using the API in R
------------------

From here, the API can be used in much the same way as in a Python script, with the caveats that Python variables sometimes need converting to R, and vice versa. For instance, a data frame can be converted using the following methods:

.. sourcecode:: r

   df <- py_to_r(df)
   # or
   df <- r_to_py(df)

Bear in mind that with Python methods which take an integer input, it is best to explicitly convert the number using ``as.integer(x)``

Python style dictionaries can be explicitly constructed using ``reticulate``'s builtins

.. sourcecode:: r

   builtins <- import_builtins
   builtins$dict(my_r_list)

Example Script
--------------

Please see the below example which loads and queries a model in R.

.. sourcecode:: r
    :linenos:

    if (!requireNamespace("reticulate", quietly = TRUE)) {
      install.packages("reticulate", repos = "https://cloud.r-project.org")
    }
    library(reticulate)

    use_condaenv("problm-solver", required = TRUE)
    py_config()
    llama <- import("problm_solver.llama_interface")
    #adj <- import("problm_solver.llama_interface")

    model <- llama$ModelInstance(
      fname = "path/to/model.gguf",
      context = "Why is the sky blue?",
      n_ctx = as.integer(4096),
      logits_all = TRUE
    )
    out <- model$query()
    cat(out, "\n")


Sampling Functions in R
-----------------------

``reticulate`` implicitly converts between Python and R callables, so to create a custom sampling function, you can define it as an R function, and use ``problm_solver``'s ``ModelInstance.generate_adjusted()`` function as usual.

Bear in mind that a sampling function takes a generic dataclass ``GenerationContext`` as an argument, so the R function will need to follow the following scheme

.. sourcecode:: r
    :linenos:

    sampling_fn <- function(context) {
      token_probs <- py_to_r(context$token_probs)
      # Modify log probabilities as needed...
      token_probs
    }

    py_sampling_fn <- r_to_py(sampling_fn)

    out <- model$generate_adjusted(
      top_k = as.integer(30),
      top_p = 0.9,
      adjust_fn = py_sampling_fn,
      max_tokens = as.integer(1024)
    )
