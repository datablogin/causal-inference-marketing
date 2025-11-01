#!/bin/bash

# Fix mixed patterns in sensitivity module
for file in libs/causal_inference/causal_inference/sensitivity/*.py; do
    # Fix patterns like: CovariateData | NDArray[Any] | Optional[pd.DataFrame]
    sed -i '' 's/CovariateData | NDArray\[Any\] | Optional\[pd\.DataFrame\]/Optional[Union[CovariateData, NDArray[Any], pd.DataFrame]]/g' "$file"
    # Fix patterns like: NDArray[Any] | Optional[pd.Series]
    sed -i '' 's/NDArray\[Any\] | Optional\[pd\.Series\]/Optional[Union[NDArray[Any], pd.Series]]/g' "$file"
    # Fix patterns like: NDArray[Any] | Optional[pd.DataFrame]
    sed -i '' 's/NDArray\[Any\] | Optional\[pd\.DataFrame\]/Optional[Union[NDArray[Any], pd.DataFrame]]/g' "$file"
done

# Fix streaming.py mixed pattern
file="libs/causal_inference/causal_inference/utils/streaming.py"
sed -i '' 's/Union\[str, Path\] | pd\.DataFrame/Union[str, Path, pd.DataFrame]/g' "$file"

echo "Fixed mixed union patterns"
