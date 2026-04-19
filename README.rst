==============================
Bayesian Feature Selection
==============================

.. image:: https://img.shields.io/pypi/v/bayesian_feature_selection.svg
        :target: https://pypi.python.org/pypi/bayesian_feature_selection

.. image:: https://github.com/MacroMagic/bayesian_feature_selection/actions/workflows/ci.yml/badge.svg
        :target: https://github.com/MacroMagic/bayesian_feature_selection/actions/workflows/ci.yml

.. image:: https://readthedocs.org/projects/bayesian-feature-selection/badge/?version=latest
        :target: https://bayesian-feature-selection.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status


Scalable Bayesian feature selection using horseshoe priors with NumPyro/JAX.
Select relevant features from high-dimensional data using a Bayesian GLM with
the regularized horseshoe prior, which provides strong shrinkage for irrelevant
features while preserving truly relevant signals.

* Free software: MIT license
* Documentation: https://bayesian-feature-selection.readthedocs.io.


Features
--------

* **Horseshoe Prior GLM** — Regularized horseshoe prior for automatic feature selection
* **Multiple GLM Families** — Gaussian (regression), Binomial (classification), and Poisson (count data)
* **MCMC & SVI Inference** — Full posterior via NUTS or fast approximation via Stochastic Variational Inference
* **Flexible Feature Selection** — Beta-based, lambda-based, or combined inclusion probabilities
* **Data Loading & Preprocessing** — Built-in CSV loading, train/test splitting, and standardization
* **CLI Interface** — Run experiments from the command line with YAML configuration files
* **Visualization** — Feature importance plots and MCMC diagnostic plots via ArviZ


Quick Start
-----------

.. code-block:: python

    import numpy as np
    from bayesian_feature_selection import HorseshoeGLM, InferenceConfig

    # Create synthetic data
    rng = np.random.RandomState(42)
    n, p = 100, 10
    X = rng.randn(n, p)
    true_beta = np.array([3.0, -2.0, 0, 0, 1.5, 0, 0, 0, 0, 0])
    y = X @ true_beta + rng.randn(n) * 0.5

    # Fit model
    model = HorseshoeGLM(family="gaussian")
    config = InferenceConfig(
        method="mcmc", num_warmup=500, num_samples=1000,
        num_chains=2, use_gpu=False, progress_bar=False
    )
    model.fit(X, y, config=config)

    # Get selected features
    importance = model.get_feature_importance(threshold=0.5)
    selected = importance[importance["selected"]]
    print(selected[["feature_idx", "beta_mean", "beta_inclusion_prob"]])

    # Make predictions
    predictions = model.predict(X)


CLI Usage
---------

.. code-block:: console

    $ bayesian-fs -c configs/default.yaml


Installation
------------

.. code-block:: console

    $ pip install bayesian_feature_selection

For development:

.. code-block:: console

    $ pip install -e ".[dev]"


Environment Setup (Python 3.12 + CUDA 12)
------------------------------------------

**Python 3.12 (CPU only)**

.. code-block:: console

    $ pip install bayesian_feature_selection

This installs JAX ≥ 0.7.0 and NumPyro ≥ 0.15.0 automatically.

**Python 3.12 + CUDA 12 (GPU)**

Install the package, then upgrade JAX with the CUDA 12 backend:

.. code-block:: console

    $ pip install bayesian_feature_selection
    $ pip install "jax[cuda12]"

Verify the setup:

.. code-block:: python

    import jax
    print(jax.devices())  # should show CudaDevice(id=0) when GPU is available

**Important version notes**

* NumPyro ≥ 0.15.0 requires JAX ≥ 0.7.0.
* JAX ≥ 0.10.0 removed internal symbols that NumPyro ≤ 0.20.1 depends on;
  use JAX 0.7.x – 0.9.x until NumPyro releases a compatible update.
* The ``gpu`` extra (``pip install bayesian_feature_selection[gpu]``) installs
  ``jax[cuda12]`` and is the recommended way to enable GPU support.


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
