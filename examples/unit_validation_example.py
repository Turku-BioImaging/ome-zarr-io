#!/usr/bin/env python3
"""
Demonstration of unit validation for space axes in OME-Zarr schema dataclasses.

This example shows how the library validates units for space-type axes
according to the OME-Zarr specification.
"""

from ome_zarr_writer.schema_models import Axis, create_cyx_axes


def demonstrate_valid_units():
    """Show examples of valid units for space axes."""
    print("Valid space units in OME-Zarr:")
    print("=" * 50)

    # Some common microscopy units
    valid_units = [
        "micrometer",  # Most common in microscopy
        "nanometer",  # High-resolution microscopy
        "millimeter",  # Larger biological samples
        "meter",  # Macroscopic samples
        "angstrom",  # Atomic-scale imaging
        "inch",  # Sometimes used in industrial imaging
        "centimeter",  # Medium-scale samples
    ]

    for unit in valid_units:
        try:
            Axis(name="y", type="space", unit=unit)
            print(f"✓ {unit:12} - Valid space unit")
        except ValueError as e:
            print(f"✗ {unit:12} - Error: {e}")

    print("\nCreating CYX axes with different units:")
    for unit in ["micrometer", "nanometer", "angstrom"]:
        axes = create_cyx_axes(0.1, 0.1, unit=unit)
        print(f"  CYX axes with {unit}: {[ax.unit for ax in axes]}")


def demonstrate_invalid_units():
    """Show examples of invalid units that will raise errors."""
    print("\nInvalid space units (will raise errors):")
    print("=" * 50)

    invalid_units = [
        "microns",  # Common mistake (should be "micrometer")
        "nm",  # Abbreviation not allowed (should be "nanometer")
        "um",  # Abbreviation not allowed (should be "micrometer")
        "pixels",  # Not a physical unit
        "mm",  # Abbreviation not allowed (should be "millimeter")
        "custom_unit",  # Arbitrary units not allowed
    ]

    for unit in invalid_units:
        try:
            Axis(name="y", type="space", unit=unit)
            print(f"✗ {unit:12} - Should have failed but didn't!")
        except ValueError as e:
            print(f"✓ {unit:12} - Correctly rejected: {str(e)[:60]}...")


def demonstrate_all_valid_units():
    """Show the complete list of valid units from the OME-Zarr specification."""
    print("\nComplete list of valid space units:")
    print("=" * 50)

    # Get the complete list from the Axis class
    all_units = set(Axis.VALID_SPACE_UNITS)

    # Group them by scale for easier reading
    length_units = {
        "Very small": [
            "yoctometer",
            "zeptometer",
            "attometer",
            "femtometer",
            "picometer",
        ],
        "Small": ["angstrom", "nanometer", "micrometer"],
        "Medium": ["millimeter", "centimeter", "decimeter", "meter"],
        "Large": [
            "kilometer",
            "megameter",
            "gigameter",
            "terameter",
            "petameter",
            "exameter",
            "zettameter",
            "yottameter",
        ],
        "Imperial": ["inch", "foot", "yard", "mile"],
        "Astronomical": ["parsec"],
    }

    for category, units in length_units.items():
        print(f"\n{category}:")
        for unit in units:
            if unit in all_units:
                print(f"  • {unit}")

    # Show any units we might have missed
    covered_units = set()
    for units in length_units.values():
        covered_units.update(units)

    missed_units = all_units - covered_units
    if missed_units:
        print("\nOther units:")
        for unit in sorted(missed_units):
            print(f"  • {unit}")


if __name__ == "__main__":
    print("OME-Zarr Space Axis Unit Validation Demo")
    print("=" * 60)

    demonstrate_valid_units()
    demonstrate_invalid_units()
    demonstrate_all_valid_units()

    print(f"\nTotal valid units: {len(Axis.VALID_SPACE_UNITS)}")
    print("\nFor more information, see the OME-Zarr specification:")
    print("https://ngff.openmicroscopy.org/latest/#axes-md")
