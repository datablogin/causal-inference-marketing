# Integration with Analytics Backend Monorepo

This document outlines how this causal inference marketing repository is designed for seamless integration with the existing `analytics-backend-monorepo`.

## Current Structure (Separate Repository)

```
causal-inference-marketing/
├── libs/
│   └── causal_inference/           # Shared library for causal inference
│       ├── causal_inference/
│       │   ├── core/              # Attribution, incrementality, MMM
│       │   ├── data/              # Data processing utilities
│       │   └── utils/             # Helper functions
│       ├── tests/
│       └── pyproject.toml
├── services/
│   ├── causal_api/                # FastAPI service for causal inference
│   └── causal_processor/          # Background job processing (planned) 
├── shared/
│   ├── config/                    # Configuration management (extracted from monorepo)
│   ├── database/                  # Database abstractions (compatible)
│   └── observability/             # Logging and metrics (compatible)
├── docker/                        # Docker configurations
└── docs/                          # Documentation
```

## Target Structure (After Monorepo Integration)

```
analytics-backend-monorepo/
├── libs/
│   ├── causal_inference/          # ← MOVE FROM separate repo
│   ├── analytics_core/            # Existing
│   ├── api_common/               # Existing
│   └── config/                   # Existing (merge shared/config/)
├── services/
│   ├── causal_api/               # ← MOVE FROM separate repo
│   ├── causal_processor/         # ← MOVE FROM separate repo
│   ├── analytics_api/            # Existing
│   └── ml_inference/             # Existing
└── ...                           # Other existing structure
```

## Integration Strategy

### Phase 1: Preparation (Complete)
- ✅ Mirror monorepo directory structure (`libs/`, `services/`, `shared/`)
- ✅ Align dependencies and Python version (3.11+)
- ✅ Match tool configurations (ruff, mypy, pytest)
- ✅ Extract and adapt shared patterns (config, database, observability)
- ✅ Create FastAPI services following monorepo patterns
- ✅ Extract GitHub infrastructure (workflows, issue templates, PR templates)
- ✅ Set up automated dependency management (Dependabot)

### Phase 2: Integration Process

#### Step 1: Move Shared Library
```bash
# In analytics-backend-monorepo:
cp -r ../causal-inference-marketing/libs/causal_inference libs/
```

#### Step 2: Move Services
```bash
# In analytics-backend-monorepo:
cp -r ../causal-inference-marketing/services/causal_api services/
cp -r ../causal-inference-marketing/services/causal_processor services/
```

#### Step 3: Merge Configuration
- Integrate `shared/config/causal_config.py` into existing `libs/config/`
- Update `libs/config/__init__.py` to include causal inference config
- Merge any custom configuration patterns

#### Step 4: Update Dependencies
- Add causal inference dependencies to main `pyproject.toml`
- Ensure version compatibility across all libraries

#### Step 5: Update CI/CD
- Add causal inference library to GitHub Actions test matrix
- Include causal inference services in deployment pipeline
- Add service-specific Docker configurations

#### Step 6: Merge GitHub Infrastructure
- Merge issue templates (adapt existing templates to include causal inference components)
- Update CI/CD workflows to include causal inference in test matrix
- Merge PR templates with causal inference specific sections
- Configure Dependabot for causal inference dependencies

#### Step 7: Update Documentation
- Integrate causal inference docs into monorepo documentation
- Update API documentation to include new endpoints

### Phase 3: Clean Integration

#### Remove Duplicated Infrastructure
Once integrated, remove duplicated infrastructure from this repository:
- Remove `shared/` directory (use monorepo versions)
- Remove duplicate configurations
- Update imports to use monorepo shared libraries

#### Update Import Paths
```python
# Before (separate repo):
from shared.config import CausalInferenceConfig
from shared.database import get_database_manager

# After (monorepo):
from libs.config import CausalInferenceConfig  
from libs.database import get_database_manager
```

## Compatibility Matrix

| Component | Separate Repo | Monorepo | Status |
|-----------|---------------|----------|---------|
| Python Version | 3.11+ | 3.11+ | ✅ Compatible |
| FastAPI | 0.104.0+ | 0.104.0+ | ✅ Compatible |
| SQLAlchemy | 2.0.0+ | 2.0.0+ | ✅ Compatible |
| Pydantic | 2.5.0+ | 2.5.0+ | ✅ Compatible |
| Ruff | 0.1.6+ | 0.1.6+ | ✅ Compatible |
| MyPy | 1.7.0+ | 1.7.0+ | ✅ Compatible |
| Pytest | 7.4.0+ | 7.4.0+ | ✅ Compatible |

## Testing Integration

### Before Integration
Run compatibility tests to ensure seamless integration:

```bash
# Test shared library
cd libs/causal_inference && python -m pytest tests/

# Test configuration compatibility  
python -c "from shared.config import CausalInferenceConfig; print('✅ Config compatible')"

# Test API service
cd services/causal_api && uvicorn main:app --port 8001
curl http://localhost:8001/health/
```

### After Integration
Verify integration works within monorepo:

```bash
# From monorepo root:
make ci-causal-inference
make test-integration
```

## Migration Checklist

- [ ] **Backup current work**: Ensure all changes are committed
- [ ] **Test compatibility**: Run all tests in separate repo
- [ ] **Copy libraries**: Move `libs/causal_inference` to monorepo
- [ ] **Copy services**: Move services to monorepo
- [ ] **Update imports**: Change import paths to use monorepo structure
- [ ] **Merge configurations**: Integrate configs with existing patterns
- [ ] **Update CI/CD**: Add causal inference to build pipeline
- [ ] **Test integration**: Verify everything works in monorepo
- [ ] **Update documentation**: Merge docs and update references
- [ ] **Clean up**: Remove duplicate infrastructure

## Benefits of Integration

### For Causal Inference Library
- ✅ Access to existing database and API infrastructure
- ✅ Shared authentication and authorization systems
- ✅ Unified observability and monitoring
- ✅ Consistent deployment and CI/CD patterns
- ✅ Integration with existing analytics services

### For Analytics Backend Monorepo
- ✅ Enhanced analytics capabilities with causal inference
- ✅ Marketing-specific attribution and incrementality testing
- ✅ Media mix modeling for comprehensive analytics
- ✅ Additional ML capabilities and Bayesian modeling
- ✅ Extended API surface for client applications

## Future Enhancements

After integration, additional enhancements become possible:
- Cross-service data sharing and pipelines
- Unified authentication across all analytics services  
- Shared feature store for ML models
- Integrated experimentation platform
- Unified analytics dashboard

This design ensures that the causal inference library can be developed independently while maintaining full compatibility for future monorepo integration.