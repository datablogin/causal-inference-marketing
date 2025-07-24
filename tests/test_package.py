"""Basic tests for package structure and imports."""


def test_package_import():
    """Test that the main package can be imported."""
    import causal_inference_marketing

    assert causal_inference_marketing.__version__ == "0.1.0"


def test_submodule_imports():
    """Test that submodules can be imported."""
    from causal_inference_marketing import core, data, utils

    # Basic import test - modules should exist
    assert core is not None
    assert data is not None
    assert utils is not None
