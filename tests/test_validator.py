"""
Tests for the OME-Zarr validator.
"""

import pytest
from jsonschema.exceptions import ValidationError
from invalid_cases import (
    SCHEMA_IMAGE_CASES,
    SCHEMA_LABEL_CASES,
    SEMANTIC_IMAGE_CASES,
    SEMANTIC_LABEL_CASES,
    read_attrs,
)
from ome_zarr_io.validator import OMEZarrValidator
from ome_zarr_io.schema_models import (
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


# -- Realistic broken metadata (see tests/invalid_cases.py) ---------------------


@pytest.fixture
def valid_image_attrs(valid_fileset):
    return read_attrs(valid_fileset)


@pytest.fixture
def valid_label_attrs(valid_fileset):
    return read_attrs(valid_fileset / "labels" / "nuclei")


def dotted_issues(validator, attrs, schema_type):
    """Issues as "<dotted.path> <message>" strings, the form used by `expected` fragments."""
    return [
        f"{'.'.join(map(str, path))} {message}"
        for path, message in validator.iter_issues(attrs, schema_type)
    ]


def test_control_written_metadata_is_valid(validator, valid_image_attrs, valid_label_attrs):
    assert validator.validate_image_metadata(valid_image_attrs) is True
    assert validator.validate_label_metadata(valid_label_attrs) is True


@pytest.mark.parametrize(
    "case_id,mutate,expected", SCHEMA_IMAGE_CASES, ids=[c[0] for c in SCHEMA_IMAGE_CASES]
)
def test_image_schema_rejects_broken_metadata(validator, valid_image_attrs, case_id, mutate, expected):
    mutate(valid_image_attrs)

    with pytest.raises(ValidationError):
        validator.validate_image_metadata(valid_image_attrs)

    errors = validator.get_validation_errors(valid_image_attrs, "image")
    assert errors
    assert any(expected in text for text in dotted_issues(validator, valid_image_attrs, "image"))
    # The strict schema is a superset of the permissive one.
    assert len(validator.get_validation_errors(valid_image_attrs, "strict_image")) >= len(errors)


@pytest.mark.parametrize(
    "case_id,mutate,expected", SCHEMA_LABEL_CASES, ids=[c[0] for c in SCHEMA_LABEL_CASES]
)
def test_label_schema_rejects_broken_metadata(validator, valid_label_attrs, case_id, mutate, expected):
    mutate(valid_label_attrs)

    with pytest.raises(ValidationError):
        validator.validate_label_metadata(valid_label_attrs)

    assert validator.get_validation_errors(valid_label_attrs, "label")
    assert any(expected in text for text in dotted_issues(validator, valid_label_attrs, "label"))


@pytest.mark.parametrize(
    "case_id,mutate,expected", SEMANTIC_IMAGE_CASES, ids=[c[0] for c in SEMANTIC_IMAGE_CASES]
)
def test_json_schema_alone_misses_semantic_image_errors(validator, valid_image_attrs, case_id, mutate, expected):
    """These are invalid, but only `ome_zarr_io.validate()` catches them (see test_report)."""
    mutate(valid_image_attrs)
    assert validator.get_validation_errors(valid_image_attrs, "image") == []


@pytest.mark.parametrize(
    "case_id,mutate,expected", SEMANTIC_LABEL_CASES, ids=[c[0] for c in SEMANTIC_LABEL_CASES]
)
def test_json_schema_alone_misses_semantic_label_errors(validator, valid_label_attrs, case_id, mutate, expected):
    mutate(valid_label_attrs)
    assert validator.get_validation_errors(valid_label_attrs, "label") == []


def test_iter_issues_returns_structured_paths(validator, valid_image_attrs):
    valid_image_attrs["ome"]["multiscales"][0]["datasets"][0]["path"] = 0

    issues = validator.iter_issues(valid_image_attrs, "image")

    assert issues == [(("ome", "multiscales", 0, "datasets", 0, "path"), "0 is not of type 'string'")]


def test_iter_issues_unknown_schema_type_raises(validator, valid_image_attrs):
    with pytest.raises(ValueError, match="Schema type 'nope' not found"):
        validator.iter_issues(valid_image_attrs, "nope")
