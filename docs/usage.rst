=====
Usage
=====

Quick Start
-----------

Import the main class, create some data, fit the model, and extract feature importance:

.. code-block:: python

    import numpy as np
    from bayesian_feature_selection import HorseshoeGLM, InferenceConfig

    # Generate synthetic data
    np.random.seed(42)
    n, p = 100, 20
    X = np.random.randn(n, p)
    true_beta = np.zeros(p)
    true_beta[:5] = [3.0, -2.0, 1.5, -1.0, 0.5]
    y = X @ true_beta + np.random.randn(n) * 0.5

    # Fit model
    model = HorseshoeGLM(family="gaussian", scale_global=1.0)
    config = InferenceConfig(method="mcmc", num_warmup=500, num_samples=1000, num_chains=1)
    model.fit(X, y, config=config)

    # Get feature importance
    importance = model.get_feature_importance(threshold=0.5, method="beta")
    print(importance[importance["selected"]])

Using DataLoader with CSV Files
-------------------------------

Load data from a CSV file using :class:`~bayesian_feature_selection.DataLoader`:

.. code-block:: python

    from bayesian_feature_selection import DataLoader, DataConfig

    data_config = DataConfig(
        data_path="data/my_dataset.csv",
        target_col="target",
        feature_cols=None,  # Use all columns except target
        standardize=True,
        test_size=0.2,
        random_seed=42,
    )

    loader = DataLoader(data_config)
    X_train, X_test, y_train, y_test, feature_names = loader.load_and_split()

MCMC vs SVI Inference
---------------------

**MCMC** (Markov Chain Monte Carlo) provides exact posterior samples but is slower:

.. code-block:: python

    mcmc_config = InferenceConfig(
        method="mcmc",
        num_warmup=1000,
        num_samples=2000,
        num_chains=4,
        use_gpu=True,
        progress_bar=True,
    )
    model.fit(X, y, config=mcmc_config)

**SVI** (Stochastic Variational Inference) is faster but provides an approximate posterior:

.. code-block:: python

    svi_config = InferenceConfig(
        method="svi",
        num_steps=10000,
        learning_rate=0.001,
        use_gpu=True,
        progress_bar=True,
    )
    model.fit(X, y, config=svi_config)

Feature Selection Methods
-------------------------

Three methods are available via ``get_feature_importance()``:

- **beta**: Based on the coefficient posterior. Selects features with consistent non-zero effects. Best for prediction and interpretation.
- **lambda**: Based on the local shrinkage parameter. Identifies features with weak shrinkage. Better for filtering pure noise.
- **both**: Combines beta and lambda inclusion probabilities.

.. code-block:: python

    # Beta-based selection (default)
    importance_beta = model.get_feature_importance(threshold=0.5, method="beta")

    # Lambda-based selection
    importance_lambda = model.get_feature_importance(threshold=0.5, method="lambda")

    # Combined selection
    importance_both = model.get_feature_importance(threshold=0.5, method="both")

CLI Usage
---------

The package provides a ``bayesian-fs`` command-line interface:

.. code-block:: console

    # Run with a YAML config file
    $ bayesian-fs -c configs/default.yaml

    # Specify output directory
    $ bayesian-fs -c configs/default.yaml -o results/experiment1

    # Override model family and inference method
    $ bayesian-fs -c configs/default.yaml --family binomial --method svi

    # Enable GPU
    $ bayesian-fs -c configs/default.yaml --use-gpu

Configuration via YAML Files
-----------------------------

Create a YAML configuration file to define all experiment parameters:

.. code-block:: yaml

    data:
      data_path: "data/my_dataset.csv"
      target_col: "target"
      feature_cols: null
      test_size: 0.2
      standardize: true
      random_seed: 42

    model:
      family: "gaussian"
      scale_global: 1.0

    inference:
      method: "mcmc"
      num_warmup: 1000
      num_samples: 2000
      num_chains: 4
      use_gpu: true
      progress_bar: true

    selection:
      method: "beta"
      threshold: 0.5

    output:
      save_plots: true
      save_diagnostics: true
      save_samples: false

Load and use the configuration programmatically:

.. code-block:: python

    from pathlib import Path
    from bayesian_feature_selection import ExperimentConfig

    config = ExperimentConfig.from_yaml(Path("configs/my_experiment.yaml"))

    # Modify a parameter
    config.inference.num_samples = 5000

    # Save the updated config
    config.to_yaml(Path("configs/updated.yaml"))

Making Predictions
------------------

After fitting, use the model to make predictions on new data:

.. code-block:: python

    # Point predictions (posterior mean)
    y_pred = model.predict(X_new)

    # Full posterior predictive samples
    y_samples = model.predict(X_new, return_samples=True)
    print(y_samples.shape)  # (num_samples, n_new)
