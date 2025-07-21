#!/usr/bin/env python3
"""
Example demonstrating time axis unit validation in OME-Zarr metadata.

This example shows how to use the create_axes function with time axis units
according to the OME-Zarr 0.5 specification.
"""

from ome_zarr_writer import create_axes, VALID_TIME_UNITS

# Example 1: Time-lapse with seconds
print("Creating time-lapse axes with seconds...")
axes_seconds = create_axes(
    axes="tyx",
    x_size=0.1,  # 0.1 micrometer pixel size
    y_size=0.1,  # 0.1 micrometer pixel size
    time_unit="second",
    unit="micrometer",
)
print(f"Time axis: {axes_seconds[0].name} (unit: {axes_seconds[0].unit})")
print(f"Y axis: {axes_seconds[1].name} (unit: {axes_seconds[1].unit})")
print(f"X axis: {axes_seconds[2].name} (unit: {axes_seconds[2].unit})")
print()

# Example 2: Long-term study with days
print("Creating long-term study axes with days...")
axes_days = create_axes(
    axes="tcyx",
    x_size=1.0,  # 1.0 millimeter pixel size
    y_size=1.0,  # 1.0 millimeter pixel size
    time_unit="day",
    unit="millimeter",
)
print(f"Time axis: {axes_days[0].name} (unit: {axes_days[0].unit})")
print(f"Channel axis: {axes_days[1].name}")
print(f"Y axis: {axes_days[2].name} (unit: {axes_days[2].unit})")
print(f"X axis: {axes_days[3].name} (unit: {axes_days[3].unit})")
print()

# Example 3: High-speed imaging with milliseconds
print("Creating high-speed imaging axes with milliseconds...")
axes_ms = create_axes(
    axes="tczyx",
    x_size=0.05,  # 0.05 nanometer pixel size
    y_size=0.05,  # 0.05 nanometer pixel size
    z_size=0.1,  # 0.1 nanometer z-step
    time_unit="millisecond",
    unit="nanometer",
)
print(f"Time axis: {axes_ms[0].name} (unit: {axes_ms[0].unit})")
print(f"Channel axis: {axes_ms[1].name}")
print(f"Z axis: {axes_ms[2].name} (unit: {axes_ms[2].unit})")
print(f"Y axis: {axes_ms[3].name} (unit: {axes_ms[3].unit})")
print(f"X axis: {axes_ms[4].name} (unit: {axes_ms[4].unit})")
print()

# Example 4: Show all supported time units
print("All supported time units:")
for unit in sorted(VALID_TIME_UNITS):
    print(f"  - {unit}")
print()

# Example 5: Error handling for invalid time units
print("Demonstrating error handling for invalid time units...")
try:
    invalid_axes = create_axes(
        axes="tyx",
        x_size=0.1,
        y_size=0.1,
        time_unit="invalid_unit",  # This will raise an error
        unit="micrometer",
    )
except ValueError as e:
    print(f"Error caught: {e}")
print()

print("Time unit validation example completed successfully!")
