========
Tutorial
========

This tutorial walks through a complete end-to-end workflow for Bayesian feature
selection using the horseshoe prior.

Step 1: Install the Package
----------------------------

Install from PyPI:

.. code-block:: console

    $ pip install bayesian_feature_selection

Or install from source for development:

.. code-block:: console

    $ git clone https://github.com/MacroMagic/bayesian_feature_selection.git
    $ cd bayesian_feature_selection
    $ pip install -e ".[dev]"

Step 2: Prepare Data
--------------------

Create a synthetic dataset with known sparse structure for demonstration:

.. code-block:: python

    import numpy as np

    np.random.seed(42)
    n_samples = 200
    n_features = 50

    # Generate random features
    X = np.random.randn(n_samples, n_features)

    # Only 5 features are truly relevant
    true_beta = np.zeros(n_features)
    true_beta[0] = 3.0
    true_beta[1] = -2.0
    true_beta[5] = 1.5
    true_beta[10] = -1.0
    true_beta[20] = 0.5

    # Generate response
    y = X @ true_beta + np.random.randn(n_samples) * 0.5

    feature_names = [f"gene_{i}" for i in range(n_features)]

Step 3: Configure the Experiment
---------------------------------

**Programmatic configuration:**

.. code-block:: python

    from bayesian_feature_selection import (
        HorseshoeGLM,
        InferenceConfig,
        ModelConfig,
        SelectionConfig,
        ExperimentConfig,
        DataConfig,
        OutputConfig,
    )

    model_config = ModelConfig(family="gaussian", scale_global=0.5)
    inference_config = InferenceConfig(
        method="mcmc",
        num_warmup=500,
        num_samples=1000,
        num_chains=2,
        use_gpu=False,
    )
    selection_config = SelectionConfig(method="beta", threshold=0.5)

**YAML configuration:**

Save the following as ``experiment.yaml``:

.. code-block:: yaml

    data:
      data_path: "data/synthetic.csv"
      target_col: "y"
      standardize: true
      test_size: 0.2

    model:
      family: "gaussian"
      scale_global: 0.5

    inference:
      method: "mcmc"
      num_warmup: 500
      num_samples: 1000
      num_chains: 2
      use_gpu: false

    selection:
      method: "beta"
      threshold: 0.5

    output:
      save_plots: true
      save_diagnostics: true

Then load it:

.. code-block:: python

    from pathlib import Path
    from bayesian_feature_selection import ExperimentConfig

    config = ExperimentConfig.from_yaml(Path("experiment.yaml"))

Step 4: Fit the Model
---------------------

**Using MCMC:**

.. code-block:: python

    model = HorseshoeGLM(
        family=model_config.family,
        scale_global=model_config.scale_global,
    )
    model.fit(X, y, config=inference_config)

**Using SVI (faster, approximate):**

.. code-block:: python

    svi_config = InferenceConfig(
        method="svi",
        num_steps=5000,
        learning_rate=0.005,
        use_gpu=False,
    )
    model_svi = HorseshoeGLM(family="gaussian", scale_global=0.5)
    model_svi.fit(X, y, config=svi_config)

Step 5: Analyze Results
-----------------------

Extract feature importance and examine the selected features:

.. code-block:: python

    importance = model.get_feature_importance(
        threshold=selection_config.threshold,
        method=selection_config.method,
    )

    # Add feature names
    importance["feature_name"] = [feature_names[i] for i in importance["feature_idx"]]

    # Show selected features
    selected = importance[importance["selected"]]
    print("Selected features:")
    print(selected[["feature_name", "beta_mean", "beta_inclusion_prob"]])

    # Show all features sorted by importance
    print("\nAll features by inclusion probability:")
    print(importance[["feature_name", "beta_mean", "beta_inclusion_prob"]].head(10))

Features with ``beta_inclusion_prob`` above the threshold are marked as
selected. The ``ci_excludes_zero`` column indicates whether the 95% credible
interval excludes zero, providing additional evidence of relevance.

Step 6: Make Predictions
------------------------

.. code-block:: python

    # Generate new data
    X_new = np.random.randn(10, n_features)

    # Point predictions
    y_pred = model.predict(X_new)
    print("Predictions:", y_pred)

    # Posterior predictive samples for uncertainty quantification
    y_samples = model.predict(X_new, return_samples=True)
    y_mean = y_samples.mean(axis=0)
    y_std = y_samples.std(axis=0)
    print("Prediction uncertainty (std):", y_std)

Step 7: Save and Visualize Results
-----------------------------------

.. code-block:: python

    from pathlib import Path
    from bayesian_feature_selection.visualization import (
        plot_feature_importance,
        plot_diagnostics,
    )

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    # Save feature importance to CSV
    importance.to_csv(output_dir / "feature_importance.csv", index=False)

    # Plot feature importance (saves PNG files)
    plot_feature_importance(importance, output_dir, feature_names=feature_names)

    # Plot MCMC diagnostics (trace plots, posterior, R-hat)
    if model.mcmc is not None:
        plot_diagnostics(model.mcmc, output_dir)

    # Save the experiment configuration for reproducibility
    experiment_config = ExperimentConfig(
        model=model_config,
        inference=inference_config,
        selection=selection_config,
    )
    experiment_config.to_yaml(output_dir / "config.yaml")
