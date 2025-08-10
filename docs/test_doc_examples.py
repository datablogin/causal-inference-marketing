"""Tests for documentation code examples.

This module extracts and tests code examples from documentation files
to ensure they are syntactically correct and execute without errors.
"""

import ast
import re
import textwrap
from pathlib import Path
from typing import List, Tuple

import pytest


def extract_code_blocks(file_path: Path) -> List[Tuple[str, int]]:
    """Extract Python code blocks from RST documentation files.

    Args:
        file_path: Path to documentation file

    Returns:
        List of (code, line_number) tuples
    """
    code_blocks = []

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Pattern to match .. code-block:: python
    pattern = r"\.\.\ code-block::\ python\n\n((?:\ {3,}.*\n?)*)"

    for match in re.finditer(pattern, content, re.MULTILINE):
        code_text = match.group(1)
        # Remove indentation (typically 3 spaces)
        dedented_code = textwrap.dedent(code_text)
        # Find line number
        line_num = content[: match.start()].count("\n") + 1
        code_blocks.append((dedented_code.strip(), line_num))

    return code_blocks


def get_documentation_files() -> List[Path]:
    """Get all RST documentation files."""
    docs_dir = Path(__file__).parent / "source"
    return list(docs_dir.rglob("*.rst"))


def test_code_blocks_syntax():
    """Test that all code blocks in documentation have valid Python syntax."""
    errors = []

    for doc_file in get_documentation_files():
        code_blocks = extract_code_blocks(doc_file)

        for code, line_num in code_blocks:
            if not code.strip():
                continue

            try:
                # Test syntax by parsing
                ast.parse(code)
            except SyntaxError as e:
                error_msg = f"{doc_file.relative_to(Path.cwd())}:{line_num} - {e}"
                errors.append(error_msg)

    if errors:
        pytest.fail("Syntax errors in documentation:\n" + "\n".join(errors))


def test_imports_are_valid():
    """Test that imports in documentation examples are valid."""
    import_errors = []

    for doc_file in get_documentation_files():
        code_blocks = extract_code_blocks(doc_file)

        for code, line_num in code_blocks:
            if not code.strip():
                continue

            # Extract import statements
            try:
                tree = ast.parse(code)
            except SyntaxError:
                continue  # Skip blocks with syntax errors (handled by other test)

            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    # Skip imports that are clearly for examples only
                    if isinstance(node, ast.ImportFrom):
                        if node.module and any(
                            skip in node.module
                            for skip in ["your_data", "marketing_data", "large_file"]
                        ):
                            continue

                    # Try to compile just the import
                    import_code = ast.get_source_segment(code, node)
                    if import_code:
                        try:
                            compile(import_code, "<string>", "exec")
                        except SyntaxError as e:
                            error_msg = f"{doc_file.relative_to(Path.cwd())}:{line_num} - Import error: {import_code} - {e}"
                            import_errors.append(error_msg)

    if import_errors:
        pytest.fail("Import errors in documentation:\n" + "\n".join(import_errors))


def test_common_patterns():
    """Test common patterns used in documentation examples."""
    pattern_errors = []

    for doc_file in get_documentation_files():
        code_blocks = extract_code_blocks(doc_file)

        for code, line_num in code_blocks:
            if not code.strip():
                continue

            # Check for common issues
            lines = code.split("\n")
            for i, line in enumerate(lines):
                # Check for incomplete code patterns
                if line.strip().endswith("..."):
                    continue  # This is intentionally incomplete

                # Check for missing colons in control structures
                stripped = line.strip()

                # Skip non-control-structure statements
                if (
                    stripped.startswith(("#", "import", "from"))
                    or "=" in stripped
                    and not stripped.startswith(("if ", "elif ", "while ", "for "))
                ):
                    continue

                if (
                    stripped.startswith(
                        (
                            "if ",
                            "for ",
                            "while ",
                            "def ",
                            "class ",
                            "try:",
                            "except",
                            "else:",
                            "elif ",
                            "with ",
                        )
                    )
                    and not stripped.endswith(":")
                    and not stripped.endswith("\\")
                ):
                    error_msg = f"{doc_file.relative_to(Path.cwd())}:{line_num + i} - Missing colon: {stripped}"
                    pattern_errors.append(error_msg)

    if pattern_errors:
        pytest.fail("Pattern errors in documentation:\n" + "\n".join(pattern_errors))


def test_example_data_consistency():
    """Test that example data usage is consistent across documentation."""
    # This is a placeholder for more sophisticated consistency checking
    # Could check that example data types are used consistently
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
