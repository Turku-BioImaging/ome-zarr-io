"""
Tests for the OME-Zarr schema dataclasses.
"""

import pytest
from ome_zarr_writer.schema_models import (
    OMEZarrImageMetadata,
    OMEMetadata,
    Multiscale,
    Dataset,
    Axis,
    ScaleTransformation,
    TranslationTransformation,
    Channel,
    Window,
    Omero,
    create_2d_axes,
    create_3d_axes,
    create_scale_transformation,
)


def test_simple_2d_metadata():
    """Test creating simple 2D metadata."""
    axes = create_2d_axes(0.1, 0.1)
    scale_transform = create_scale_transformation([0.1, 0.1])

    dataset = Dataset(path="0", coordinateTransformations=[scale_transform])

    multiscale = Multiscale(datasets=[dataset], axes=axes)

    ome_metadata = OMEMetadata(multiscales=[multiscale], version="0.5")

    metadata = OMEZarrImageMetadata(ome=ome_metadata)

    # Should not raise any exceptions
    attrs_dict = metadata.to_dict()

    # Basic validation
    assert "ome" in attrs_dict
    assert "multiscales" in attrs_dict["ome"]
    assert "version" in attrs_dict["ome"]
    assert attrs_dict["ome"]["version"] == "0.5"


def test_multiscale_3d_metadata():
    """Test creating 3D multiscale metadata."""
    axes = create_3d_axes(0.1, 0.1, 0.2)

    datasets = []
    for level in range(3):
        scale_factor = 2**level
        scale_transform = ScaleTransformation(
            scale=[0.2 * scale_factor, 0.1 * scale_factor, 0.1 * scale_factor]
        )

        dataset = Dataset(path=str(level), coordinateTransformations=[scale_transform])
        datasets.append(dataset)

    multiscale = Multiscale(datasets=datasets, axes=axes)

    ome_metadata = OMEMetadata(multiscales=[multiscale], version="0.5")

    metadata = OMEZarrImageMetadata(ome=ome_metadata)

    attrs_dict = metadata.to_dict()

    # Check we have multiple datasets
    assert len(attrs_dict["ome"]["multiscales"][0]["datasets"]) == 3

    # Check scales are correctly set
    first_scale = attrs_dict["ome"]["multiscales"][0]["datasets"][0][
        "coordinateTransformations"
    ][0]["scale"]
    assert first_scale == [0.2, 0.1, 0.1]

    second_scale = attrs_dict["ome"]["multiscales"][0]["datasets"][1][
        "coordinateTransformations"
    ][0]["scale"]
    assert second_scale == [0.4, 0.2, 0.2]


def test_omero_metadata():
    """Test creating metadata with OMERO settings."""
    axes = [
        Axis(name="c", type="channel"),
        Axis(name="y", type="space", unit="micrometer"),
        Axis(name="x", type="space", unit="micrometer"),
    ]

    scale_transform = ScaleTransformation(scale=[1.0, 0.1, 0.1])
    dataset = Dataset(path="0", coordinateTransformations=[scale_transform])

    channels = [
        Channel(
            label="DAPI",
            color="0000FF",
            window=Window(start=0, min=0, end=4095, max=4095),
            active=True,
        ),
        Channel(
            label="GFP",
            color="00FF00",
            window=Window(start=100, min=0, end=3000, max=4095),
            active=True,
        ),
    ]

    omero = Omero(channels=channels)

    multiscale = Multiscale(datasets=[dataset], axes=axes)
    ome_metadata = OMEMetadata(multiscales=[multiscale], version="0.5", omero=omero)
    metadata = OMEZarrImageMetadata(ome=ome_metadata)

    attrs_dict = metadata.to_dict()

    # Check OMERO is present
    assert "omero" in attrs_dict["ome"]
    assert "channels" in attrs_dict["ome"]["omero"]
    assert len(attrs_dict["ome"]["omero"]["channels"]) == 2

    # Check first channel
    ch1 = attrs_dict["ome"]["omero"]["channels"][0]
    assert ch1["label"] == "DAPI"
    assert ch1["color"] == "0000FF"
    assert ch1["active"] is True
    assert ch1["window"]["start"] == 0
    assert ch1["window"]["end"] == 4095


def test_validation_errors():
    """Test that validation catches errors."""

    # Empty datasets should fail
    with pytest.raises(ValueError, match="At least one dataset is required"):
        Multiscale(datasets=[], axes=create_2d_axes(0.1, 0.1))

    # Scale array too short should fail
    with pytest.raises(ValueError, match="Scale array must have at least 2 elements"):
        ScaleTransformation(scale=[0.1])

    # Translation array too short should fail
    with pytest.raises(
        ValueError, match="Translation array must have at least 2 elements"
    ):
        TranslationTransformation(translation=[0.1])

    # Insufficient axes count should fail first
    axes = [Axis(name="x", type="space", unit="micrometer")]
    dataset = Dataset(
        path="0", coordinateTransformations=[ScaleTransformation(scale=[0.1, 0.1])]
    )
    with pytest.raises(ValueError, match="Axes must have between 2 and 5 items"):
        Multiscale(datasets=[dataset], axes=axes)

    # Missing scale transformation should fail at Dataset level
    axes = create_2d_axes(0.1, 0.1)
    with pytest.raises(
        ValueError, match="At least one scale transformation is required"
    ):
        Dataset(
            path="0",
            coordinateTransformations=[
                TranslationTransformation(translation=[0.0, 0.0])
            ],
        )


def test_roundtrip_conversion():
    """Test converting to dict and back."""
    # Create metadata
    axes = create_2d_axes(0.1, 0.1)
    scale_transform = create_scale_transformation([0.1, 0.1])
    dataset = Dataset(path="0", coordinateTransformations=[scale_transform])
    multiscale = Multiscale(datasets=[dataset], axes=axes)
    ome_metadata = OMEMetadata(multiscales=[multiscale], version="0.5")
    original = OMEZarrImageMetadata(ome=ome_metadata)

    # Convert to dict and back
    data_dict = original.to_dict()
    reconstructed = OMEZarrImageMetadata.from_dict(data_dict)
    reconstructed_dict = reconstructed.to_dict()

    # Should be identical
    assert data_dict == reconstructed_dict


def test_translation_and_scale():
    """Test dataset with both scale and translation."""
    axes = create_2d_axes(0.1, 0.1)

    transformations = [
        ScaleTransformation(scale=[0.1, 0.1]),
        TranslationTransformation(translation=[100.0, 50.0]),
    ]

    dataset = Dataset(path="0", coordinateTransformations=transformations)
    multiscale = Multiscale(datasets=[dataset], axes=axes)
    ome_metadata = OMEMetadata(multiscales=[multiscale], version="0.5")
    metadata = OMEZarrImageMetadata(ome=ome_metadata)

    attrs_dict = metadata.to_dict()

    # Check both transformations are present
    coord_transforms = attrs_dict["ome"]["multiscales"][0]["datasets"][0][
        "coordinateTransformations"
    ]
    assert len(coord_transforms) == 2

    # Find scale and translation
    scale_transform = next(t for t in coord_transforms if t["type"] == "scale")
    translation_transform = next(
        t for t in coord_transforms if t["type"] == "translation"
    )

    assert scale_transform["scale"] == [0.1, 0.1]
    assert translation_transform["translation"] == [100.0, 50.0]


def test_axis_uniqueness():
    """Test that axis names must be unique."""
    axes = [
        Axis(name="c", type="channel"),  # Duplicate channel
        Axis(name="c", type="channel"),  # Duplicate channel
        Axis(name="y", type="space", unit="micrometer"),
        Axis(name="x", type="space", unit="micrometer"),
    ]

    dataset = Dataset(
        path="0",
        coordinateTransformations=[ScaleTransformation(scale=[1.0, 1.0, 0.1, 0.1])],
    )

    with pytest.raises(ValueError, match="Axis names must be unique"):
        Multiscale(datasets=[dataset], axes=axes)
