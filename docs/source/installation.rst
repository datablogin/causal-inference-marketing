Installation Guide
==================

This guide covers different ways to install Causal Inference Marketing Tools based on your use case.

Prerequisites
-------------

- Python 3.9 or higher
- pip or uv package manager

System Requirements
-------------------

**Minimum Requirements:**
- Python 3.9+
- 4GB RAM
- 1GB disk space

**Recommended for Production:**
- Python 3.11+
- 16GB RAM
- 10GB disk space

Installation Options
====================

Core Installation
-----------------

For basic causal inference functionality:

.. code-block:: bash

   pip install causal-inference-marketing

This installs the core dependencies:

- numpy, pandas, scipy
- scikit-learn, statsmodels
- pydantic for data validation
- FastAPI for the API service

With Machine Learning Extensions
--------------------------------

For advanced ML-based estimators:

.. code-block:: bash

   pip install "causal-inference-marketing[ml]"

Additional dependencies:

- lightgbm for gradient boosting
- shap for explainability
- mlflow for experiment tracking
- alibi for counterfactual explanations

With Documentation Tools
------------------------

To build documentation locally:

.. code-block:: bash

   pip install "causal-inference-marketing[docs]"

Development Installation
========================

For contributors and developers:

.. code-block:: bash

   git clone https://github.com/datablogin/causal-inference-marketing
   cd causal-inference-marketing

   # Using make (recommended)
   make install-dev

   # Or manually with uv
   uv pip install -e ".[dev,test,ml,docs]"

This installs all dependencies including:

- Development tools (ruff, mypy, pytest)
- Testing frameworks
- Documentation builders
- ML extensions

Docker Installation
===================

Using Docker Compose:

.. code-block:: bash

   git clone https://github.com/datablogin/causal-inference-marketing
   cd causal-inference-marketing
   docker-compose up -d

This starts:
- FastAPI service on port 8000
- PostgreSQL database on port 5432
- Prometheus monitoring on port 9090

Verification
============

Verify your installation:

.. code-block:: python

   import causal_inference
   from causal_inference.estimators import GComputation
   from causal_inference.core import TreatmentData

   print(f"Version: {causal_inference.__version__}")

   # Test basic functionality
   estimator = GComputation()
   print("✅ Installation successful!")

Troubleshooting
===============

Common Issues
-------------

**ImportError: No module named 'causal_inference'**

Make sure you're using the correct package name:

.. code-block:: python

   # Correct
   from causal_inference.estimators import GComputation

   # Not: from causal_inference_marketing import ...

**Memory Issues with Large Datasets**

For large datasets (>1M rows), consider:

.. code-block:: bash

   # Install with performance optimizations
   pip install "causal-inference-marketing[ml]"

   # Use memory-efficient options
   export OMP_NUM_THREADS=1
   export OPENBLAS_NUM_THREADS=1

**Windows-Specific Issues**

On Windows, some dependencies may require Visual Studio Build Tools:

1. Install Visual Studio Build Tools
2. Install with verbose output to see any compilation issues:

.. code-block:: bash

   pip install -v causal-inference-marketing

Getting Help
============

If you encounter installation issues:

1. Check our `FAQ <../faq.html>`_
2. Search existing `GitHub Issues <https://github.com/datablogin/causal-inference-marketing/issues>`_
3. Create a new issue with:
   - Your Python version (``python --version``)
   - Your OS and version
   - Complete error traceback

Environment Setup
=================

Development Environment
-----------------------

For the best development experience:

.. code-block:: bash

   # Clone and set up development environment
   git clone https://github.com/datablogin/causal-inference-marketing
   cd causal-inference-marketing

   # Create virtual environment (optional but recommended)
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install in development mode
   make install-dev

   # Verify development setup
   make ci

Production Environment
----------------------

For production deployments:

.. code-block:: bash

   # Install core + ML dependencies
   pip install "causal-inference-marketing[ml]"

   # Set environment variables
   export CAUSAL_INFERENCE_DB_URL="postgresql://..."
   export CAUSAL_INFERENCE_LOG_LEVEL="INFO"

   # Start API service
   python -m services.causal_api.main

Next Steps
==========

After installation:

1. Try the :doc:`quickstart` tutorial
2. Explore :doc:`tutorials/first_analysis`
3. Read the :doc:`methodology/method_selection` guide
