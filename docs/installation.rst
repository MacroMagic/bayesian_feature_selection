.. highlight:: shell

============
Installation
============


Stable release
--------------

To install bayesian feature selection, run this command in your terminal:

.. code-block:: console

    $ pip install bayesian_feature_selection

This is the preferred method to install bayesian feature selection, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for bayesian feature selection can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/MacroMagic/bayesian_feature_selection

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/MacroMagic/bayesian_feature_selection/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ pip install .


Development installation
------------------------

To install the package in development mode with all dev dependencies:

.. code-block:: console

    $ git clone git://github.com/MacroMagic/bayesian_feature_selection
    $ cd bayesian_feature_selection
    $ pip install -e ".[dev]"

This installs the package in editable mode so that changes to the source code
take effect immediately.


GPU support
-----------

To enable GPU acceleration (requires a CUDA-compatible GPU and drivers):

.. code-block:: console

    $ pip install -e ".[gpu]"

This installs the GPU-enabled version of JAX. Verify GPU availability in Python:

.. code-block:: python

    import jax
    print(jax.devices())  # Should list GPU devices


.. _Github repo: https://github.com/MacroMagic/bayesian_feature_selection
.. _tarball: https://github.com/MacroMagic/bayesian_feature_selection/tarball/master
