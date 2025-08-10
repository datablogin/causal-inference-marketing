Causal Inference Marketing Tools Documentation
============================================

.. image:: https://img.shields.io/badge/python-3.9+-blue.svg
   :target: https://www.python.org/downloads/

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: https://github.com/datablogin/causal-inference-marketing/blob/main/LICENSE

Welcome to the comprehensive documentation for **Causal Inference Marketing Tools**,
a production-ready library for causal inference methods specifically designed for
marketing and business applications.

🚀 Quick Navigation
===================

.. grid:: 2

    .. grid-item-card::  🎯 Get Started
        :link: quickstart
        :link-type: doc

        New to causal inference? Start here for a 5-minute introduction
        and your first analysis.

    .. grid-item-card::  📚 API Reference
        :link: api/index
        :link-type: doc

        Complete documentation of all classes, methods, and functions.

    .. grid-item-card::  🔬 Methodology Guide
        :link: methodology/index
        :link-type: doc

        When to use which method, statistical assumptions, and theory.

    .. grid-item-card::  🛠 Tutorials
        :link: tutorials/index
        :link-type: doc

        Step-by-step guides for real-world business scenarios.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   installation
   quickstart
   tutorials/first_analysis

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   user_guide/overview
   user_guide/data_preparation
   user_guide/estimator_selection
   user_guide/diagnostics

.. toctree::
   :maxdepth: 2
   :caption: Methodology
   :hidden:

   methodology/index
   methodology/estimator_guide
   methodology/assumptions
   methodology/method_selection

.. toctree::
   :maxdepth: 2
   :caption: Tutorials & Examples
   :hidden:

   tutorials/index
   tutorials/marketing_attribution
   tutorials/incrementality_testing
   tutorials/media_mix_modeling
   examples/notebooks

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api/index
   api/estimators
   api/diagnostics
   api/data_models
   api/utils

.. toctree::
   :maxdepth: 2
   :caption: Production Deployment
   :hidden:

   deployment/index
   deployment/fastapi_guide
   deployment/docker_setup
   deployment/monitoring

.. toctree::
   :maxdepth: 2
   :caption: Help & Support
   :hidden:

   troubleshooting/index
   troubleshooting/common_errors
   troubleshooting/debugging_guide
   faq
   contributing

What is Causal Inference?
========================

Causal inference is the process of determining cause-and-effect relationships from data.
In marketing, this helps answer questions like:

- **Attribution**: Which marketing channels are driving conversions?
- **Incrementality**: What is the lift from our advertising campaigns?
- **Media Mix Modeling**: How should we allocate our marketing budget?
- **A/B Testing**: What is the true treatment effect of our experiments?

Key Features
============

🎯 **Marketing-Focused**
   - Pre-built estimators for common marketing use cases
   - Real-world examples with marketing data
   - Production-ready FastAPI service

🔬 **Comprehensive Methods**
   - G-computation (Standardization)
   - Inverse Probability Weighting (IPW)
   - Augmented IPW (Doubly Robust)
   - Double Machine Learning (DML)
   - Instrumental Variables
   - Regression Discontinuity
   - Synthetic Control

📊 **Robust Diagnostics**
   - Assumption testing
   - Sensitivity analysis
   - Balance diagnostics
   - Effect visualization

🚀 **Production Ready**
   - Type-safe Pydantic models
   - Comprehensive error handling
   - Monitoring and observability
   - Scalable architecture

Installation
============

**Quick Install (Core)**

.. code-block:: bash

   pip install causal-inference-marketing

**With ML Dependencies**

.. code-block:: bash

   pip install "causal-inference-marketing[ml]"

**Development Install**

.. code-block:: bash

   git clone https://github.com/datablogin/causal-inference-marketing
   cd causal-inference-marketing
   make install-dev

Quick Example
=============

Here's a simple example of estimating the causal effect of a marketing channel:

.. code-block:: python

   import pandas as pd
   from causal_inference.estimators import GComputation
   from causal_inference.core import TreatmentData, OutcomeData, CovariateData

   # Load your data
   data = pd.read_csv("marketing_data.csv")

   # Define your causal problem
   treatment = TreatmentData(
       values=data["email_campaign"],
       name="email_campaign"
   )
   outcome = OutcomeData(
       values=data["conversion"],
       name="conversion"
   )
   covariates = CovariateData(
       values=data[["age", "previous_purchases", "website_visits"]],
       names=["age", "previous_purchases", "website_visits"]
   )

   # Estimate causal effect
   estimator = GComputation()
   effect = estimator.estimate_ate(
       treatment=treatment,
       outcome=outcome,
       covariates=covariates
   )

   print(f"Average Treatment Effect: {effect.ate:.3f}")
   print(f"95% CI: [{effect.ci_lower:.3f}, {effect.ci_upper:.3f}]")
   print(f"P-value: {effect.p_value:.3f}")

Next Steps
==========

1. **New Users**: Start with the :doc:`quickstart` guide
2. **Choose Your Method**: Read the :doc:`methodology/method_selection` guide
3. **Learn by Example**: Try our :doc:`tutorials/index`
4. **Production Deployment**: Set up the :doc:`deployment/fastapi_guide`

Community & Support
===================

- 📖 **Documentation**: https://causal-inference-marketing.readthedocs.io
- 🐛 **Bug Reports**: https://github.com/datablogin/causal-inference-marketing/issues
- 💬 **Discussions**: https://github.com/datablogin/causal-inference-marketing/discussions
- 📧 **Email**: causal-inference@yourcompany.com

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
