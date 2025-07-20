"""Test configuration and fixtures."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil


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
