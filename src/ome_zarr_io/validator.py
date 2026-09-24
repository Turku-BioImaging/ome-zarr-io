"""JSON Schema validation for OME-Zarr metadata."""

import json
from pathlib import Path
from typing import Any, Dict
from jsonschema import Draft202012Validator, RefResolver
from jsonschema.exceptions import ValidationError


class OMEZarrValidator:
    """Validates OME-Zarr metadata against JSON schemas."""

    def __init__(self, schema_version: str = "0.5"):
        """Initialize the validator with a specific schema version.

        Args:
            schema_version: The OME-Zarr schema version to use (default: "0.5")
        """
        self.schema_version = schema_version
        self._schemas = {}
        self._resolver = None
        self._load_schemas()

    def _load_schemas(self) -> None:
        """Load all schema files and create a resolver."""
        schema_path = (
            Path(__file__).parent.parent / "spec" / self.schema_version / "schemas"
        )

        # Load all schema files
        for schema_file in schema_path.glob("*.schema"):
            with open(schema_file, "r") as f:
                schema = json.load(f)
                schema_id = schema.get("$id")
                if schema_id:
                    self._schemas[schema_id] = schema

        # Create a resolver for handling $ref links
        self._resolver = RefResolver(
            base_uri=f"https://ngff.openmicroscopy.org/{self.schema_version}/schemas/",
            referrer=self._schemas.get(
                f"https://ngff.openmicroscopy.org/{self.schema_version}/schemas/ome_zarr.schema",
                {},
            ),
            store=self._schemas,
        )

    def validate_metadata(
        self, metadata: Dict[str, Any], schema_type: str = "ome_zarr"
    ) -> bool:
        """Validate metadata against a specific schema.

        Args:
            metadata: The metadata to validate
            schema_type: The type of schema to validate against
                        (e.g., "image", "plate", "well", "label", "ome_zarr")

        Returns:
            True if validation passes

        Raises:
            ValidationError: If validation fails
            ValueError: If schema type is not found
        """
        schema_id = f"https://ngff.openmicroscopy.org/{self.schema_version}/schemas/{schema_type}.schema"

        if schema_id not in self._schemas:
            raise ValueError(f"Schema type '{schema_type}' not found")

        schema = self._schemas[schema_id]
        validator = Draft202012Validator(schema, resolver=self._resolver)

        # Validate and raise detailed error if validation fails
        try:
            validator.validate(metadata)
            return True
        except ValidationError as e:
            # Re-raise with more context
            raise ValidationError(
                f"Validation failed for schema '{schema_type}': {e.message}"
            ) from e

    def validate_image_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Validate image metadata specifically.

        Args:
            metadata: The zarr.json attributes containing OME metadata

        Returns:
            True if validation passes
        """
        return self.validate_metadata(metadata, "image")

    def validate_plate_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Validate plate metadata specifically.

        Args:
            metadata: The zarr.json attributes containing OME metadata

        Returns:
            True if validation passes
        """
        return self.validate_metadata(metadata, "plate")

    def validate_well_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Validate well metadata specifically.

        Args:
            metadata: The zarr.json attributes containing OME metadata

        Returns:
            True if validation passes
        """
        return self.validate_metadata(metadata, "well")

    def validate_label_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Validate label metadata specifically.

        Args:
            metadata: The zarr.json attributes containing OME metadata

        Returns:
            True if validation passes
        """
        return self.validate_metadata(metadata, "label")

    def get_validation_errors(
        self, metadata: Dict[str, Any], schema_type: str = "ome_zarr"
    ) -> list:
        """Get all validation errors without raising an exception.

        Args:
            metadata: The metadata to validate
            schema_type: The type of schema to validate against

        Returns:
            List of validation error messages
        """
        schema_id = f"https://ngff.openmicroscopy.org/{self.schema_version}/schemas/{schema_type}.schema"

        if schema_id not in self._schemas:
            return [f"Schema type '{schema_type}' not found"]

        schema = self._schemas[schema_id]
        validator = Draft202012Validator(schema, resolver=self._resolver)

        errors = []
        for error in validator.iter_errors(metadata):
            errors.append(
                f"Path: {' -> '.join(str(p) for p in error.absolute_path)}, Error: {error.message}"
            )

        return errors
