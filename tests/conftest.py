"""Test configuration and fixtures."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU/CUDA hardware"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically skip GPU tests when CUDA/CuPy is not available."""
    # Check if CuPy is available and if we can detect CUDA devices
    try:
        import cupy
        # Try to get device count to ensure CUDA is functional
        device_count = cupy.cuda.runtime.getDeviceCount()
        cuda_available = device_count > 0
    except (ImportError, Exception):
        cuda_available = False
    
    # Skip GPU tests if CUDA is not available
    if not cuda_available:
        skip_gpu = pytest.mark.skip(reason="CUDA/CuPy not available or no GPU devices found")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    return np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
