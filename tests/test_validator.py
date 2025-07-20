"""Tests for OME-Zarr schema validation."""

import pytest
from ome_zarr_writer.validator import OMEZarrValidator
from jsonschema.exceptions import ValidationError


class TestOMEZarrValidator:
    """Test cases for the OME-Zarr validator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = OMEZarrValidator()
    
    def test_validator_initialization(self):
        """Test that validator initializes correctly."""
        assert self.validator.schema_version == "0.5"
        assert self.validator._schemas is not None
        assert self.validator._resolver is not None
    
    def test_valid_image_metadata(self):
        """Test validation of valid image metadata."""
        valid_metadata = {
            "ome": {
                "version": "0.5",
                "multiscales": [
                    {
                        "version": "0.5",
                        "name": "test_image",
                        "axes": [
                            {"name": "y", "type": "space", "unit": "micrometer"},
                            {"name": "x", "type": "space", "unit": "micrometer"}
                        ],
                        "datasets": [
                            {
                                "path": "0",
                                "coordinateTransformations": [
                                    {
                                        "type": "scale",
                                        "scale": [1.0, 1.0]
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        }
        
        # Should not raise an exception
        assert self.validator.validate_image_metadata(valid_metadata) is True
    
    def test_invalid_image_metadata(self):
        """Test validation of invalid image metadata."""
        invalid_metadata = {
            "ome": {
                "version": "0.5"
                # Missing required multiscales field
            }
        }
        
        # Should raise ValidationError
        with pytest.raises(ValidationError):
            self.validator.validate_image_metadata(invalid_metadata)
    
    def test_get_validation_errors(self):
        """Test getting validation errors without raising exceptions."""
        invalid_metadata = {
            "ome": {
                "version": "0.5"
                # Missing required multiscales field
            }
        }
        
        errors = self.validator.get_validation_errors(invalid_metadata, "image")
        assert len(errors) > 0
        assert any("multiscales" in error for error in errors)
    
    def test_unknown_schema_type(self):
        """Test validation with unknown schema type."""
        metadata = {"test": "data"}
        
        with pytest.raises(ValueError, match="Schema type 'unknown' not found"):
            self.validator.validate_metadata(metadata, "unknown")
    
    def test_different_schema_types(self):
        """Test that different validation methods work."""
        # This is a basic test - in reality you'd need valid metadata for each type
        test_metadata = {"ome": {"version": "0.5"}}
        
        # These should all raise ValidationError due to missing required fields
        with pytest.raises(ValidationError):
            self.validator.validate_image_metadata(test_metadata)
        
        with pytest.raises(ValidationError):
            self.validator.validate_plate_metadata(test_metadata)
        
        with pytest.raises(ValidationError):
            self.validator.validate_well_metadata(test_metadata)
        
        with pytest.raises(ValidationError):
            self.validator.validate_label_metadata(test_metadata)
