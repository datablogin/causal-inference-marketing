#!/usr/bin/env python3
"""Fix all Python 3.10+ union syntax to be compatible with Python 3.9."""

import re
import sys
from pathlib import Path


def fix_union_syntax_comprehensive(content: str) -> tuple[str, bool]:
    """Replace all X | Y with Union[X, Y] and ensure Union is imported."""
    modified = False

    # Pattern to match type unions (A | B) but not bitwise operations
    # This matches patterns in type annotations (after : or ->)
    patterns = [
        # Return type annotations: ) -> Type1 | Type2
        (r"(\) *-> *)([^\n:]+\|[^\n:]+)(\s*:)", r"\1Union[\2]\3"),
        # Return type annotations at end of line: ) -> Type1 | Type2
        (r"(\) *-> *)([^\n]+\|[^\n]+)(\s*$)", r"\1Union[\2]\3"),
    ]

    new_content = content
    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, new_content, flags=re.MULTILINE)
        if result != new_content:
            modified = True
            new_content = result

    if not modified:
        return content, False

    # Clean up the Union syntax by removing extra spaces around |
    new_content = re.sub(r"\| +", ", ", new_content)
    new_content = re.sub(r" +\|", ", ", new_content)

    # Check if Union is already imported
    has_union_import = re.search(r"from typing import.*\bUnion\b", new_content)

    if not has_union_import:
        # Find the typing import line and add Union
        typing_import_pattern = r"(from typing import )(.*?)(\n)"
        match = re.search(typing_import_pattern, new_content)

        if match:
            # Add Union to existing typing import
            imports = match.group(2)
            if "Union" not in imports:
                new_imports = imports.rstrip() + ", Union"
                new_content = re.sub(
                    typing_import_pattern,
                    r"\g<1>" + new_imports + r"\g<3>",
                    new_content,
                    count=1,
                )

    return new_content, True


def process_file(file_path: Path) -> bool:
    """Process a single file and return True if modified."""
    try:
        content = file_path.read_text()
        new_content, modified = fix_union_syntax_comprehensive(content)

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
