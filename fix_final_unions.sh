#!/bin/bash

# Fix sensitivity module
for file in libs/causal_inference/causal_inference/sensitivity/*.py; do
    if ! grep -q "from typing import.*Union" "$file" 2>/dev/null; then
        sed -i '' 's/^from typing import \(.*\)/from typing import \1, Union/' "$file" 2>/dev/null || true
    fi
    # Fix multi-type unions in function parameters
    sed -i '' 's/: TreatmentData | NDArray\[Any\] | pd\.Series/: Union[TreatmentData, NDArray[Any], pd.Series]/g' "$file"
    sed -i '' 's/: OutcomeData | NDArray\[Any\] | pd\.Series/: Union[OutcomeData, NDArray[Any], pd.Series]/g' "$file"
    sed -i '' 's/: NDArray\[Any\] | list\[float\]/: Union[NDArray[Any], list[float]]/g' "$file"
    sed -i '' 's/: NDArray\[Any\] | pd\.Series | pd\.DataFrame/: Union[NDArray[Any], pd.Series, pd.DataFrame]/g' "$file"
    sed -i '' 's/: NDArray\[Any\] | pd\.Series/: Union[NDArray[Any], pd.Series]/g' "$file"
done

# Fix utils/memory_efficient.py
file="libs/causal_inference/causal_inference/utils/memory_efficient.py"
sed -i '' 's/: pd\.DataFrame | NDArray\[Any\]/: Union[pd.DataFrame, NDArray[Any]]/g' "$file"
sed -i '' 's/: Callable\[\[pd\.DataFrame | NDArray\[Any\]\]/: Callable[[Union[pd.DataFrame, NDArray[Any]]]/g' "$file"
sed -i '' 's/pd\.DataFrame | NDArray\[Any\]\]/Union[pd.DataFrame, NDArray[Any]]]/g' "$file"
sed -i '' 's/: NDArray\[Any\] | sparse\.spmatrix/: Union[NDArray[Any], sparse.spmatrix]/g' "$file"

# Fix utils/streaming.py
file="libs/causal_inference/causal_inference/utils/streaming.py"
if ! grep -q "from typing import.*Union" "$file"; then
    sed -i '' 's/from typing import \(.*\)/from typing import \1, Union/' "$file"
fi
sed -i '' 's/: str | Path/: Union[str, Path]/g' "$file"
sed -i '' 's/: str | Path | pd\.DataFrame/: Union[str, Path, pd.DataFrame]/g' "$file"

# Fix utils/validation.py
file="libs/causal_inference/causal_inference/utils/validation.py"
if ! grep -q "from typing import.*Union" "$file"; then
    sed -i '' 's/from typing import \(.*\)/from typing import \1, Union/' "$file"
fi
sed -i '' 's/: NDArray\[Any\] | pd\.Series/: Union[NDArray[Any], pd.Series]/g' "$file"

echo "Fixed final union syntax issues"
