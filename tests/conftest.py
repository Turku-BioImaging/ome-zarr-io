"""Test configuration and fixtures."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil


def pytest_configure(config):
    """Configures custom pytest markers."""
    
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU/CUDA hardware"
    )


def pytest_collection_modifyitems(config, items):
    """Checks if CuPy library and CUDA-capable GPU devices are available. If not, GPU tests will be skipped."""
    try:
        import cupy
        device_count = cupy.cuda.runtime.getDeviceCount()
        cuda_available = device_count > 0
    except (ImportError, Exception):
        cuda_available = False
    
    if not cuda_available:
        skip_gpu = pytest.mark.skip(reason="CuPy not available or no CUDA-capable GPU devices found")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


@pytest.fixture
def temp_dir():
    """Creates a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_image():
    """Creates a sample image for testing."""
    return np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
