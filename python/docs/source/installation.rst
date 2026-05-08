Installation
============

Requirements
------------

* Python ``>= 3.10``
* `pytometry <https://github.com/scverse/pytometry>`_ / ``readfcs``
  for FCS I/O (pulled in automatically)
* ``numpy``, ``scipy``, ``pandas``, ``scikit-learn``, ``matplotlib``

From PyPI
---------

The package is distributed from PyPI as ``peacoqc``:

.. code-block:: bash

    pip install peacoqc                 # core
    pip install "peacoqc[fcs]"          # + flowio, needed for write_fcs
    pip install "peacoqc[plotting]"     # + seaborn for richer plots
    pip install "peacoqc[test]"         # + pytest for running the test suite

Extras at a glance
------------------

+--------------+-------------------------------------------------------------+
| Extra        | Purpose                                                     |
+==============+=============================================================+
| ``fcs``      | Install ``flowio`` so :func:`peacoqc.write_fcs` can write   |
|              | cleaned :class:`~anndata.AnnData` objects back to FCS.      |
+--------------+-------------------------------------------------------------+
| ``plotting`` | Install ``seaborn`` for the optional richer plotting        |
|              | backend.                                                    |
+--------------+-------------------------------------------------------------+
| ``test``     | Install ``pytest`` and ``pytest-mpl`` to run the package    |
|              | test suite locally.                                         |
+--------------+-------------------------------------------------------------+
| ``docs``     | Install Sphinx, ``furo``, ``nbsphinx`` and ``myst-parser``  |
|              | to build this documentation site.                           |
+--------------+-------------------------------------------------------------+

From source
-----------

Clone the repository and install in editable mode:

.. code-block:: bash

    git clone https://github.com/mdmanurung/peacoqc.git
    cd peacoqc
    pip install -e "python/[fcs,test]"

Building the docs
-----------------

To build this documentation site locally:

.. code-block:: bash

    sudo apt-get update && sudo apt-get install -y pandoc
    pip install -e "python/[docs]"
    sphinx-build -b html python/docs/source python/docs/_build/html
    # then open python/docs/_build/html/index.html

The user guide notebook (``python/docs/vignette.ipynb``) is
pre-executed, so ``nbsphinx`` renders its embedded figures without
re-running the pipeline. ``pandoc`` is still required so the notebook
can be converted during the Sphinx build.

Verifying the installation
--------------------------

.. code-block:: python

    import peacoqc
    print(peacoqc.__version__)

    # Run against the example FCS file shipped with the upstream package.
    adata = peacoqc.read_fcs("inst/extdata/111_Comp_Trans.fcs")
    result = peacoqc.peaco_qc(
        adata,
        channels=[adata.var_names[i] for i in [0, 2, 4, 5, 6, 7]],
    )
    print(f"{result.percentage_removed:.2f}% removed")
