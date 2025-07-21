"""
Tests for the OME-Zarr validator.
"""

import pytest
from jsonschema.exceptions import ValidationError
from ome_zarr_writer.validator import OMEZarrValidator
from ome_zarr_writer.schema_models import (
    OMEZarrImageMetadata,
    OMEMetadata,
    Multiscale,
    Dataset,
    ScaleTransformation,
    create_2d_axes,
)


@pytest.fixture
def validator():
    """Create a validator instance for testing."""
    return OMEZarrValidator()


@pytest.fixture
def valid_image_metadata():
    """Create valid image metadata for testing."""
    axes = create_2d_axes(0.1, 0.1)
    scale_transform = ScaleTransformation(scale=[0.1, 0.1])
    dataset = Dataset(path="0", coordinateTransformations=[scale_transform])
    multiscale = Multiscale(datasets=[dataset], axes=axes)
    ome_metadata = OMEMetadata(multiscales=[multiscale], version="0.5")
    metadata = OMEZarrImageMetadata(ome=ome_metadata)
    return metadata.to_dict()


def test_validator_initialization(validator):
    """Test that the validator initializes correctly."""
    assert validator.schema_version == "0.5"
    assert len(validator._schemas) > 0


def test_validate_valid_image_metadata(validator, valid_image_metadata):
    """Test validation of valid image metadata."""
    result = validator.validate_image_metadata(valid_image_metadata)
    assert result is True


def test_validate_invalid_metadata_missing_ome(validator):
    """Test validation fails for metadata missing OME section."""
    invalid_metadata = {"invalid": "data"}

    with pytest.raises(ValidationError):
        validator.validate_image_metadata(invalid_metadata)


def test_validate_invalid_metadata_missing_version(validator):
    """Test validation fails for metadata missing version."""
    invalid_metadata = {"ome": {"multiscales": []}}

    with pytest.raises(ValidationError):
        validator.validate_image_metadata(invalid_metadata)


def test_validate_invalid_metadata_empty_multiscales(validator):
    """Test validation fails for empty multiscales."""
    invalid_metadata = {"ome": {"multiscales": [], "version": "0.5"}}

    with pytest.raises(ValidationError):
        validator.validate_image_metadata(invalid_metadata)


def test_get_validation_errors_returns_list(validator, valid_image_metadata):
    """Test that get_validation_errors returns empty list for valid metadata."""
    errors = validator.get_validation_errors(valid_image_metadata, "image")
    assert isinstance(errors, list)
    assert len(errors) == 0


def test_get_validation_errors_returns_errors_for_invalid(validator):
    """Test that get_validation_errors returns errors for invalid metadata."""
    invalid_metadata = {"invalid": "data"}
    errors = validator.get_validation_errors(invalid_metadata, "image")
    assert isinstance(errors, list)
    assert len(errors) > 0


def test_validate_unknown_schema_type(validator, valid_image_metadata):
    """Test that unknown schema type raises ValueError."""
    with pytest.raises(ValueError, match="Schema type 'unknown' not found"):
        validator.validate_metadata(valid_image_metadata, "unknown")


def test_get_validation_errors_unknown_schema_type(validator, valid_image_metadata):
    """Test that get_validation_errors handles unknown schema type."""
    errors = validator.get_validation_errors(valid_image_metadata, "unknown")
    assert len(errors) == 1
    assert "Schema type 'unknown' not found" in errors[0]


def test_validate_image_metadata_convenience_method(validator, valid_image_metadata):
    """Test the convenience method for image validation."""
    result = validator.validate_image_metadata(valid_image_metadata)
    assert result is True


def test_schema_dataclasses_integration_with_validator(validator):
    """Test that metadata created with dataclasses validates successfully."""
    # Create metadata using dataclasses
    axes = create_2d_axes(0.1, 0.1)
    scale_transform = ScaleTransformation(scale=[0.1, 0.1])
    dataset = Dataset(path="0", coordinateTransformations=[scale_transform])
    multiscale = Multiscale(datasets=[dataset], axes=axes, name="Test Image")
    ome_metadata = OMEMetadata(multiscales=[multiscale], version="0.5")
    metadata = OMEZarrImageMetadata(ome=ome_metadata)

    # Convert to dict and validate
    attrs_dict = metadata.to_dict()
    result = validator.validate_image_metadata(attrs_dict)
    assert result is True

    # Check that no validation errors are present
    errors = validator.get_validation_errors(attrs_dict, "image")
    assert len(errors) == 0
