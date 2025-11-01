#!/bin/bash

# Fix visualization files
for file in libs/causal_inference/causal_inference/visualization/weight_diagnostics.py \
            libs/causal_inference/causal_inference/visualization/residual_analysis.py \
            libs/causal_inference/causal_inference/visualization/propensity_plots.py; do
    # Add Union to imports if not present
    if ! grep -q "from typing import.*Union" "$file"; then
        sed -i '' 's/from typing import \(.*\)/from typing import \1, Union/' "$file"
    fi
    # Fix plt.Figure | go.Figure
    sed -i '' 's/) -> plt\.Figure | go\.Figure:/) -> Union[plt.Figure, go.Figure]:/g' "$file"
    # Fix tuple[plt.Figure | go.Figure, ...
    sed -i '' 's/tuple\[plt\.Figure | go\.Figure,/tuple[Union[plt.Figure, go.Figure],/g' "$file"
done

# Fix discovery/base.py
file="libs/causal_inference/causal_inference/discovery/base.py"
if ! grep -q "from typing import.*Union" "$file"; then
    sed -i '' 's/from typing import \(.*\)/from typing import \1, Union/' "$file"
fi
sed -i '' 's/: str | int)/: Union[str, int])/g' "$file"
sed -i '' 's/from_var: str | int,/from_var: Union[str, int],/g' "$file"
sed -i '' 's/to_var: str | int)/to_var: Union[str, int])/g' "$file"

# Fix utils/memory_efficient.py
file="libs/causal_inference/causal_inference/utils/memory_efficient.py"
if ! grep -q "from typing import.*Union" "$file"; then
    sed -i '' 's/from typing import \(.*\)/from typing import \1, Union/' "$file"
fi
sed -i '' 's/) -> NDArray\[Any\] | float:/) -> Union[NDArray[Any], float]:/g' "$file"

# Fix transportability files
for file in libs/causal_inference/causal_inference/transportability/integration.py \
            libs/causal_inference/causal_inference/transportability/weighting.py; do
    if ! grep -q "from typing import.*Union" "$file"; then
        sed -i '' 's/from typing import \(.*\)/from typing import \1, Union/' "$file"
    fi
    sed -i '' 's/data: pd\.DataFrame | NDArray\[Any\])/data: Union[pd.DataFrame, NDArray[Any]])/g' "$file"
done

# Fix transportability/tmtl.py
file="libs/causal_inference/causal_inference/transportability/tmtl.py"
if ! grep -q "from typing import.*Union" "$file"; then
    sed -i '' 's/from typing import \(.*\)/from typing import \1, Union/' "$file"
fi
sed -i '' 's/) -> dict\[str, NDArray\[Any\] | bool | int\]:/) -> dict[str, Union[NDArray[Any], bool, int]]:/g' "$file"

# Fix estimators/time_varying.py
file="libs/causal_inference/causal_inference/estimators/time_varying.py"
if ! grep -q "from typing import.*Union" "$file"; then
    sed -i '' 's/from typing import \(.*\)/from typing import \1, Union/' "$file"
fi
sed -i '' 's/time: int | str)/time: Union[int, str])/g' "$file"
sed -i '' 's/) -> dict\[int | str, float\]:/) -> dict[Union[int, str], float]:/g' "$file"

# Fix api/unified_estimator.py
file="libs/causal_inference/causal_inference/api/unified_estimator.py"
if ! grep -q "from typing import.*Union" "$file"; then
    sed -i '' 's/from typing import \(.*\)/from typing import \1, Union/' "$file"
fi
sed -i '' 's/data: pd\.DataFrame | str)/data: Union[pd.DataFrame, str])/g' "$file"

# Fix data/nhefs.py
file="libs/causal_inference/causal_inference/data/nhefs.py"
if ! grep -q "from typing import.*Union" "$file"; then
    sed -i '' 's/from typing import \(.*\)/from typing import \1, Union/' "$file"
fi
sed -i '' 's/) -> tuple\[TreatmentData, OutcomeData, CovariateData\] | pd\.DataFrame:/) -> Union[tuple[TreatmentData, OutcomeData, CovariateData], pd.DataFrame]:/g' "$file"

# Fix data/missing_data.py
file="libs/causal_inference/causal_inference/data/missing_data.py"
if ! grep -q "from typing import.*Union" "$file"; then
    sed -i '' 's/from typing import \(.*\)/from typing import \1, Union/' "$file"
fi
sed -i '' 's/) -> SimpleImputer | KNNImputer | IterativeImputer:/) -> Union[SimpleImputer, KNNImputer, IterativeImputer]:/g' "$file"

echo "Fixed all remaining union syntax issues"
