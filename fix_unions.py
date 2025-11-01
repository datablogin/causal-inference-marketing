#!/usr/bin/env python3
"""Fix Python 3.10+ union syntax to be compatible with Python 3.9."""

import re
import sys
from pathlib import Path


def fix_union_syntax(content: str) -> tuple[str, bool]:
    """Replace X | None with Optional[X] and ensure Optional is imported."""
    modified = False

    # Pattern to match type | None (but not in strings)
    # This matches various patterns like:
    # - int | None
    # - str | None
    # - SomeClass[Type] | None
    # - NDArray[np.floating[Any]] | None
    pattern = r"(\b[\w\[\], \.]+)\s*\|\s*None"

    # Check if content has union syntax
    if not re.search(pattern, content):
        return content, False

    # Replace X | None with Optional[X]
    new_content = re.sub(pattern, r"Optional[\1]", content)
    modified = new_content != content

    if not modified:
        return content, False

    # Check if Optional is already imported
    has_optional_import = re.search(r"from typing import.*\bOptional\b", new_content)

    if not has_optional_import:
        # Find the typing import line and add Optional
        typing_import_pattern = r"(from typing import )(.*?)(\n)"
        match = re.search(typing_import_pattern, new_content)

        if match:
            # Add Optional to existing typing import
            imports = match.group(2)
            if "Optional" not in imports:
                new_imports = imports.rstrip() + ", Optional"
                new_content = re.sub(
                    typing_import_pattern,
                    r"\g<1>" + new_imports + r"\g<3>",
                    new_content,
                    count=1,
                )

    return new_content, modified


def process_file(file_path: Path) -> bool:
    """Process a single file and return True if modified."""
    try:
        content = file_path.read_text()
        new_content, modified = fix_union_syntax(content)

        if modified:
            file_path.write_text(new_content)
            print(f"Fixed: {file_path}")
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return False


def main():
    """Process all Python files in libs/causal_inference/causal_inference."""
    base_path = Path("libs/causal_inference/causal_inference")

    if not base_path.exists():
        print(f"Path {base_path} does not exist", file=sys.stderr)
        sys.exit(1)

    python_files = list(base_path.rglob("*.py"))
    modified_count = 0

    for file_path in python_files:
        if process_file(file_path):
            modified_count += 1

    print(f"\nProcessed {len(python_files)} files, modified {modified_count} files")


if __name__ == "__main__":
    main()
