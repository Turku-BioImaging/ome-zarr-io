"""Example of using OME-Zarr JSON schema validation."""

from ome_zarr_writer import OMEZarrValidator


def main():
    """Demonstrate JSON schema validation for OME-Zarr metadata."""

    # Initialize the validator
    validator = OMEZarrValidator()

    # Example 1: Valid image metadata
    valid_image_metadata = {
        "ome": {
            "version": "0.5",
            "multiscales": [
                {
                    "version": "0.5",
                    "name": "example_image",
                    "axes": [
                        {"name": "y", "type": "space", "unit": "micrometer"},
                        {"name": "x", "type": "space", "unit": "micrometer"},
                    ],
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [
                                {"type": "scale", "scale": [1.0, 1.0]}
                            ],
                        }
                    ],
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [0.5, 0.5]}
                    ],
                }
            ],
        }
    }

    # Example 2: Invalid image metadata (missing required fields)
    invalid_image_metadata = {
        "ome": {
            "version": "0.5",
            # Missing multiscales - this should cause validation to fail
        }
    }

    print("=== OME-Zarr JSON Schema Validation Examples ===\\n")

    # Test valid metadata
    print("1. Testing valid image metadata:")
    try:
        is_valid = validator.validate_image_metadata(valid_image_metadata)
        print(f"   ✓ Validation passed: {is_valid}")
    except Exception as e:
        print(f"   ✗ Validation failed: {e}")

    print()

    # Test invalid metadata
    print("2. Testing invalid image metadata:")
    try:
        is_valid = validator.validate_image_metadata(invalid_image_metadata)
        print(f"   ✓ Validation passed: {is_valid}")
    except Exception as e:
        print(f"   ✗ Validation failed: {e}")

    print()

    # Get validation errors without raising exception
    print("3. Getting validation errors for invalid metadata:")
    errors = validator.get_validation_errors(invalid_image_metadata, "image")
    if errors:
        print("   Validation errors found:")
        for error in errors:
            print(f"   - {error}")
    else:
        print("   No validation errors found")

    print()

    # Example of validating against different schema types
    print("4. Available schema types for validation:")
    schema_types = ["image", "plate", "well", "label", "ome_zarr"]
    for schema_type in schema_types:
        print(f"   - {schema_type}")

    print()
    print("=== Usage in your code ===")
    print(
        """
# Import the validator
from ome_zarr_writer import OMEZarrValidator

# Create validator instance
validator = OMEZarrValidator()

# Validate your metadata
try:
    validator.validate_image_metadata(your_metadata)
    print("Metadata is valid!")
except ValidationError as e:
    print(f"Validation failed: {e}")

# Or get errors without exceptions
errors = validator.get_validation_errors(your_metadata, "image")
if not errors:
    print("Metadata is valid!")
else:
    print("Validation errors:", errors)
    """
    )


if __name__ == "__main__":
    main()
