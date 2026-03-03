===========
Methodology
===========

This page describes the statistical methodology behind the Bayesian feature
selection approach implemented in this package.

The Horseshoe Prior
-------------------

The horseshoe prior (Carvalho et al., 2010) is a continuous shrinkage prior
designed for sparse estimation in high-dimensional settings. For a coefficient
vector :math:`\beta`, the horseshoe prior is defined as:

.. math::

    \beta_j \mid \lambda_j, \tau \sim \mathcal{N}(0, \lambda_j^2 \tau^2)

    \lambda_j \sim \text{Half-Cauchy}(0, 1)

    \tau \sim \text{Half-Cauchy}(0, \tau_0)

where:

- :math:`\tau` is the **global shrinkage parameter** controlling the overall level of sparsity.
- :math:`\lambda_j` is the **local shrinkage parameter** for each feature :math:`j`, allowing individual features to escape shrinkage.
- :math:`\tau_0` is the **scale of the global prior** (``scale_global`` in the code).

The key property of the horseshoe is that its density has an infinitely tall
spike at zero (encouraging shrinkage of noise features) and heavy tails
(allowing signal features to remain large).

Regularized Horseshoe Prior
----------------------------

This package implements the **regularized horseshoe** (Piironen & Vehtari, 2017),
which adds a slab component to prevent excessively large coefficients:

.. math::

    \tilde{\lambda}_j = \sqrt{\frac{c^2 \lambda_j^2}{c^2 + \tau^2 \lambda_j^2}}

    \beta_j \mid \tilde{\lambda}_j, \tau \sim \mathcal{N}(0, \tau^2 \tilde{\lambda}_j^2)

    c^2 \sim \text{Inverse-Gamma}(1, 1)

The slab variance :math:`c^2` controls the maximum magnitude of the
coefficients. When :math:`\lambda_j` is small, :math:`\tilde{\lambda}_j \approx
\lambda_j` (strong shrinkage). When :math:`\lambda_j` is large,
:math:`\tilde{\lambda}_j \approx c / \tau` (bounded by the slab). Using an
``Inverse-Gamma(1, 1)`` prior on :math:`c^2` allows the slab width to be
learned from the data.

GLM Families
------------

The horseshoe prior is combined with a generalized linear model (GLM). Three
families are supported:

**Gaussian** (linear regression):

.. math::

    y_i \sim \mathcal{N}(\eta_i, \sigma^2), \quad \sigma \sim \text{Half-Normal}(1)

where :math:`\eta_i = \alpha + X_i \beta` is the linear predictor.

**Binomial** (logistic regression):

.. math::

    y_i \sim \text{Bernoulli}\!\left(\text{logit}^{-1}(\eta_i)\right)

**Poisson** (count regression):

.. math::

    y_i \sim \text{Poisson}\!\left(\exp(\eta_i)\right)

Feature Selection Criteria
--------------------------

After fitting the model, features are selected based on posterior inclusion
probabilities. Three methods are available:

**Beta-based selection** (``method="beta"``):
    Computes the fraction of posterior samples where :math:`|\beta_j| > 0.01`.
    Features with inclusion probability above the threshold are selected. This
    method captures both the direction and magnitude of effects.

**Lambda-based selection** (``method="lambda"``):
    Uses the local shrinkage parameters :math:`\lambda_j`. Features whose
    :math:`\lambda_j` values consistently exceed the median across all features
    are considered relevant. This method is better at filtering pure noise
    without requiring strong coefficient effects.

**Combined selection** (``method="both"``):
    Averages the beta-based and lambda-based inclusion probabilities. A feature
    must show evidence from both criteria to be selected.

Choosing ``scale_global``
--------------------------

The ``scale_global`` parameter (:math:`\tau_0`) controls the prior expected
level of sparsity. A recommended rule of thumb (Piironen & Vehtari, 2017) is:

.. math::

    \tau_0 = \frac{p_0}{p - p_0} \cdot \frac{1}{\sqrt{n}}

where:

- :math:`p_0` is the expected number of relevant features,
- :math:`p` is the total number of features, and
- :math:`n` is the number of observations.

Guidelines:

- **Sparse problems** (:math:`p_0 \ll p`): Use smaller values (0.1–0.5).
- **Moderate sparsity**: Use values around 0.5–1.0.
- **Dense problems** (:math:`p_0 \approx p`): Use larger values (1.0–2.0).

References
----------

- Carvalho, C. M., Polson, N. G., & Scott, J. G. (2010). "The horseshoe
  estimator for sparse signals." *Biometrika*, 97(2), 465–480.
- Piironen, J., & Vehtari, A. (2017). "Sparsity information and regularization
  in the horseshoe and other shrinkage priors." *Electronic Journal of
  Statistics*, 11(2), 5018–5051.
