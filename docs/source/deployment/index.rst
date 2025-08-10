Production Deployment Guide
===========================

Deploy causal inference capabilities in production environments.

.. toctree::
   :maxdepth: 2

   fastapi_guide
   docker_setup
   monitoring

Overview
--------

The library provides multiple deployment options:

- **FastAPI Web Service** - REST API for causal inference
- **Docker Containers** - Containerized deployment
- **Python Library** - Direct integration in applications
- **Jupyter Service** - Interactive analysis environment

Architecture
============

Production deployment typically includes:

- **API Gateway** - Route requests and handle authentication
- **Application Server** - FastAPI service running causal inference
- **Database** - Store results and configuration
- **Monitoring** - Track performance and errors
- **Queue System** - Handle long-running analysis jobs

.. code-block:: none

   ┌─────────────┐    ┌──────────────┐    ┌─────────────┐
   │   Client    │───→│  API Gateway │───→│  FastAPI    │
   │ Application │    │   (nginx)    │    │   Service   │
   └─────────────┘    └──────────────┘    └─────────────┘
                                                   │
                                          ┌─────────────┐
                                          │ PostgreSQL  │
                                          │  Database   │
                                          └─────────────┘

Quick Start
===========

**1. Using Docker Compose (Recommended)**

.. code-block:: bash

   git clone https://github.com/datablogin/causal-inference-marketing
   cd causal-inference-marketing
   docker-compose up -d

**2. Direct FastAPI Deployment**

.. code-block:: bash

   # Install with all dependencies
   pip install "causal-inference-marketing[ml,docs]"

   # Start the service
   cd services/causal_api
   uvicorn main:app --host 0.0.0.0 --port 8000

**3. Python Library Integration**

.. code-block:: python

   from causal_inference.estimators import AIPW
   from causal_inference.core import TreatmentData, OutcomeData, CovariateData

   # Integrate directly in your application
   estimator = AIPW()
   effect = estimator.estimate_ate(treatment, outcome, covariates)

Deployment Options
==================

FastAPI Service
---------------

**Features:**
- REST API endpoints for all estimators
- Async processing for large datasets
- Built-in validation and error handling
- OpenAPI/Swagger documentation
- Health checks and monitoring endpoints

**Endpoints:**
- ``POST /attribution/analyze`` - Run causal analysis
- ``GET /health`` - Service health check
- ``GET /docs`` - Interactive API documentation

See :doc:`fastapi_guide` for detailed setup instructions.

Docker Deployment
-----------------

**Benefits:**
- Consistent environment across deployments
- Easy scaling with orchestration platforms
- Includes all dependencies and configurations
- Support for multi-stage builds

**Images Available:**
- ``causal-api`` - FastAPI service
- ``causal-jupyter`` - Jupyter notebook environment
- ``causal-worker`` - Background job processing

See :doc:`docker_setup` for container configuration.

Kubernetes Deployment
---------------------

**Helm Chart Available** (coming soon):

.. code-block:: bash

   # Install Helm chart
   helm repo add causal-inference https://charts.causal-inference.com
   helm install causal-api causal-inference/causal-api

**Features:**
- Auto-scaling based on CPU/memory usage
- Load balancing across multiple pods
- Health checks and automatic restarts
- ConfigMap-based configuration
- Persistent volume for data storage

Cloud Deployment
================

AWS
---

**AWS ECS Fargate** (Recommended):

.. code-block:: yaml

   # task-definition.json
   {
     "family": "causal-inference-api",
     "networkMode": "awsvpc",
     "requiresCompatibilities": ["FARGATE"],
     "cpu": "1024",
     "memory": "4096",
     "containerDefinitions": [{
       "name": "causal-api",
       "image": "your-repo/causal-inference-api:latest",
       "portMappings": [{
         "containerPort": 8000,
         "protocol": "tcp"
       }],
       "environment": [
         {"name": "DATABASE_URL", "value": "postgresql://..."},
         {"name": "LOG_LEVEL", "value": "INFO"}
       ]
     }]
   }

**AWS Lambda** (for lightweight usage):

.. code-block:: python

   import json
   from causal_inference.estimators import AIPW

   def lambda_handler(event, context):
       # Parse request data
       data = json.loads(event['body'])

       # Run causal analysis
       estimator = AIPW()
       effect = estimator.estimate_ate(
           treatment=data['treatment'],
           outcome=data['outcome'],
           covariates=data['covariates']
       )

       return {
           'statusCode': 200,
           'body': json.dumps({
               'ate': effect.ate,
               'confidence_interval': [effect.ci_lower, effect.ci_upper],
               'p_value': effect.p_value
           })
       }

Google Cloud Platform
---------------------

**Cloud Run** (Serverless containers):

.. code-block:: bash

   # Deploy to Cloud Run
   gcloud run deploy causal-api \
     --image gcr.io/your-project/causal-inference-api \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --memory 4Gi \
     --cpu 2

**GKE** (Managed Kubernetes):

.. code-block:: yaml

   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: causal-api
   spec:
     replicas: 3
     selector:
       matchLabels:
         app: causal-api
     template:
       metadata:
         labels:
           app: causal-api
       spec:
         containers:
         - name: causal-api
           image: gcr.io/your-project/causal-inference-api
           ports:
           - containerPort: 8000
           resources:
             requests:
               memory: "2Gi"
               cpu: "1000m"
             limits:
               memory: "4Gi"
               cpu: "2000m"

Azure
-----

**Container Instances**:

.. code-block:: bash

   az container create \
     --resource-group myResourceGroup \
     --name causal-api \
     --image your-registry.azurecr.io/causal-inference-api:latest \
     --cpu 2 \
     --memory 4 \
     --restart-policy Always \
     --ports 8000

Configuration
=============

Environment Variables
--------------------

**Database:**
- ``DATABASE_URL`` - PostgreSQL connection string
- ``DATABASE_POOL_SIZE`` - Connection pool size (default: 10)

**API:**
- ``API_HOST`` - Host to bind to (default: 0.0.0.0)
- ``API_PORT`` - Port to bind to (default: 8000)
- ``API_WORKERS`` - Number of worker processes (default: 1)

**Logging:**
- ``LOG_LEVEL`` - Logging level (DEBUG, INFO, WARNING, ERROR)
- ``LOG_FORMAT`` - Log format (json, text)

**Monitoring:**
- ``PROMETHEUS_ENABLED`` - Enable Prometheus metrics (true/false)
- ``PROMETHEUS_PORT`` - Prometheus metrics port (default: 9090)

**Security:**
- ``API_KEY_REQUIRED`` - Require API keys (true/false)
- ``ALLOWED_ORIGINS`` - CORS allowed origins
- ``MAX_REQUEST_SIZE`` - Maximum request size in bytes

Configuration Files
-------------------

**config/production.yaml**:

.. code-block:: yaml

   database:
     url: ${DATABASE_URL}
     pool_size: 20
     max_overflow: 10

   api:
     host: "0.0.0.0"
     port: 8000
     workers: 4
     timeout: 300

   logging:
     level: "INFO"
     format: "json"
     handlers:
       - console
       - file

   monitoring:
     prometheus:
       enabled: true
       port: 9090
     health_checks:
       database: true
       memory: true
       disk: true

Scaling Considerations
======================

Horizontal Scaling
-----------------

**Load Balancing:**
- Use nginx or cloud load balancers
- Sticky sessions not required (stateless API)
- Health check endpoint: ``/health``

**Auto-scaling Metrics:**
- CPU utilization > 70%
- Memory utilization > 80%
- Request latency > 5 seconds
- Queue depth > 100 requests

Vertical Scaling
---------------

**Memory Requirements:**
- Base: 2GB per worker process
- Large datasets: 4-8GB per worker
- ML models: Additional 1-2GB

**CPU Requirements:**
- Base: 1 CPU core per worker
- ML-intensive workloads: 2-4 CPU cores per worker

Performance Optimization
=======================

Caching
-------

**Model Caching:**

.. code-block:: python

   # Cache fitted models to avoid re-training
   from causal_inference.utils.caching import ModelCache

   cache = ModelCache(max_size=100)

   # Models automatically cached by data hash
   estimator = AIPW(cache=cache)

**Result Caching:**

.. code-block:: python

   # Cache analysis results
   from causal_inference.utils.caching import ResultCache

   cache = ResultCache(backend='redis', ttl=3600)
   effect = estimator.estimate_ate(treatment, outcome, covariates, cache=cache)

Database Optimization
--------------------

**Connection Pooling:**

.. code-block:: python

   # Configure connection pool
   DATABASE_CONFIG = {
       'pool_size': 20,
       'max_overflow': 30,
       'pool_timeout': 30,
       'pool_recycle': 3600
   }

**Query Optimization:**
- Index frequently queried columns
- Use read replicas for analysis queries
- Partition large tables by date/region

Monitoring & Observability
==========================

Health Checks
------------

**Endpoint: ``GET /health``**

.. code-block:: json

   {
     "status": "healthy",
     "timestamp": "2024-01-15T10:30:00Z",
     "version": "0.1.0",
     "checks": {
       "database": "healthy",
       "memory_usage": "67%",
       "disk_usage": "45%"
     }
   }

Metrics
-------

**Prometheus Metrics:**
- ``http_requests_total`` - Total HTTP requests
- ``http_request_duration_seconds`` - Request latency
- ``analysis_duration_seconds`` - Analysis computation time
- ``model_cache_hits`` - Model cache hit rate
- ``active_analyses`` - Currently running analyses

**Custom Business Metrics:**
- Analysis requests per method type
- Effect size distributions
- Error rates by analysis type

Logging
-------

**Structured Logging:**

.. code-block:: json

   {
     "timestamp": "2024-01-15T10:30:00Z",
     "level": "INFO",
     "service": "causal-api",
     "request_id": "req_abc123",
     "method": "POST",
     "endpoint": "/attribution/analyze",
     "duration": 2.34,
     "status": 200,
     "treatment_type": "binary",
     "sample_size": 10000,
     "estimator": "AIPW",
     "effect_size": 0.123
   }

Security
========

Authentication
--------------

**API Key Authentication:**

.. code-block:: python

   # Add to request headers
   headers = {
       'X-API-Key': 'your-api-key',
       'Content-Type': 'application/json'
   }

**JWT Tokens:**

.. code-block:: python

   # OAuth 2.0 / JWT integration
   headers = {
       'Authorization': 'Bearer your-jwt-token',
       'Content-Type': 'application/json'
   }

Input Validation
---------------

**Request Size Limits:**
- Maximum request size: 100MB
- Maximum array length: 1M rows
- Timeout: 5 minutes per request

**Data Validation:**
- All inputs validated with Pydantic models
- Sanitization of file uploads
- Rate limiting: 100 requests/minute per API key

Network Security
---------------

- **HTTPS only** in production
- **CORS configuration** for web applications
- **VPC/subnet isolation** in cloud deployments
- **Web Application Firewall (WAF)** for public APIs

Backup & Disaster Recovery
==========================

Data Backup
-----------

**Database Backups:**
- Daily automated backups
- Point-in-time recovery capability
- Cross-region replication for critical data

**Model Artifacts:**
- Version control for model configurations
- Backup of trained model weights
- Recovery procedures for model failures

High Availability
-----------------

**Multi-Region Deployment:**
- Active-passive failover
- Database replication
- Health monitoring and automatic failover

**Zero-Downtime Deployments:**
- Blue-green deployment strategy
- Rolling updates for containers
- Database migration strategies

Troubleshooting
===============

Common Production Issues
-----------------------

**High Memory Usage:**
- Monitor for memory leaks in long-running analyses
- Implement request timeouts
- Use streaming for large datasets

**Slow Response Times:**
- Profile analysis performance
- Optimize database queries
- Implement result caching

**Model Convergence Failures:**
- Implement retry logic with exponential backoff
- Use more robust algorithms for edge cases
- Provide meaningful error messages to users

Performance Monitoring
---------------------

**Key Metrics to Track:**
- Average response time by endpoint
- 95th percentile response times
- Error rate by analysis type
- Resource utilization (CPU, memory, disk)

**Alerting Thresholds:**
- Response time > 10 seconds
- Error rate > 5%
- Memory usage > 90%
- Disk usage > 85%

See :doc:`monitoring` for detailed monitoring setup instructions.
