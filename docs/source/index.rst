Causal Inference Tools for Marketing Applications
==============================================

Welcome to the documentation for Causal Inference Tools for Marketing Applications.

This library provides production-ready implementations of causal inference methods
specifically designed for marketing use cases.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/index

Installation
------------

.. code-block:: bash

   pip install causal-inference-marketing

Quick Start
-----------

.. code-block:: python

   from causal_inference_marketing import Attribution
   
   # Multi-touch attribution analysis
   attribution = Attribution(method="doubly_robust")
   results = attribution.fit(data, treatment_col="channel", outcome_col="conversion")

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`