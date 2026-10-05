import re

import dask.array as da
import numpy as np
import pytest

from ome_zarr_io import Reader, Writer, random_colors, validate
from ome_zarr_io.schema_models import Channel, Omero, Window
from ome_zarr_io.channels import auto_contrast, normalize_color

UNITS = {"z": "micrometer", "y": "micrometer", "x": "micrometer"}
HEX = re.compile(r"^[0-9A-F]{6}$")


def make(tmp_path, data, dims="czyx", **kw):
    dims = list(dims)
    units = {d: "micrometer" for d in dims if d in "zyx"}
    if "t" in dims:
        units["t"] = "second"
    w = Writer(tmp_path / "img.zarr", data, dims, units, **kw)
    w.write()
    return tmp_path / "img.zarr"


def data(c=2, dtype=np.uint16):
    rng = np.random.default_rng(0)
    return rng.integers(10, 1000, size=(c, 2, 16, 16)).astype(dtype)


def channels_of(path):
    return Reader(path)._metadata.ome.omero.channels


# --- auto_contrast -------------------------------------------------------
def test_auto_contrast_hot_pixel_lowers_end():
    a = np.random.default_rng(1).integers(100, 4000, 100_000).astype(np.uint16)
    a[0] = 60000
    start, end = auto_contrast(a)
    assert end < 60000 and start >= 100


def test_auto_contrast_ignores_black_background():
    a = np.zeros(100_000, dtype=np.uint16)
    a[:20_000] = np.random.default_rng(2).integers(1000, 2000, 20_000)
    start, _ = auto_contrast(a)
    assert start >= 1000 - 5 or start == 0  # background bin dropped by limit


def test_auto_contrast_constant_and_nan():
    assert auto_contrast(np.full(10, 5, dtype=np.uint8)) == (5.0, 5.0)
    assert auto_contrast(np.full(10, np.nan, dtype=np.float32)) is None
    a = np.array([np.nan, 1.0, 2.0, 3.0], dtype=np.float32)
    assert auto_contrast(a) is not None


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_auto_contrast_within_range(dtype):
    a = data(1, dtype)
    start, end = auto_contrast(a)
    assert a.min() <= start <= end <= a.max()


# --- colors --------------------------------------------------------------
def test_random_colors_format_seed_and_prefix():
    c = random_colors(6)
    assert all(HEX.match(x) for x in c)
    assert random_colors(6) == c
    assert random_colors(3) == c[:3]
    assert random_colors(6, seed=1) != c


def test_random_colors_avoid():
    fixed = random_colors(1)[0]
    assert fixed not in random_colors(4, avoid=[fixed])


def test_normalize_color():
    assert normalize_color("#00ff00") == "00FF00"
    for bad in ("12345", "GGGGGG", 5):
        with pytest.raises(ValueError):
            normalize_color(bad)


# --- Writer --------------------------------------------------------------
def test_dict_channels_min_max_and_window(tmp_path):
    a = data()
    p = make(tmp_path, a, channels={"DAPI": {"color": "#0000ff"}, "GFP": {}})
    ch = channels_of(p)
    assert [c.label for c in ch] == ["DAPI", "GFP"]
    assert ch[0].color == "0000FF"
    assert HEX.match(ch[1].color) and ch[1].color != "0000FF"
    for i, c in enumerate(ch):
        assert c.window.min == a[i].min() and c.window.max == a[i].max()
        assert c.window.min <= c.window.start <= c.window.end <= c.window.max
    assert validate(p).is_valid


@pytest.mark.parametrize("dims,axis", [("tczyx", 1), ("czyx", 0)])
def test_min_max_per_channel_any_c_position(tmp_path, dims, axis):
    shape = [3 if d == "c" else 4 for d in dims]
    shape[-2:] = [16, 16]
    a = np.random.default_rng(3).integers(0, 500, shape).astype(np.uint16)
    p = make(tmp_path, a, dims, channels=["a", "b", "c"], colors="random")
    for i, c in enumerate(channels_of(p)):
        sub = np.take(a, i, axis=axis)
        assert (c.window.min, c.window.max) == (sub.min(), sub.max())
        assert HEX.match(c.color)


def test_dask_input(tmp_path):
    a = data()
    p = make(tmp_path, da.from_array(a, chunks=(1, 2, 8, 8)), channels=["a", "b"])
    assert channels_of(p)[1].window.max == a[1].max()


def test_window_forms(tmp_path):
    a = data(4)
    p = make(
        tmp_path,
        a,
        channels=[
            {"label": "a", "window": "minmax"},
            {"label": "b", "window": (0, 5000)},
            {"label": "c", "window": None},
            {"label": "d", "window": Window(start=1, min=0, end=2, max=3)},
        ],
    )
    a_, b_, c_, d_ = channels_of(p)
    assert (a_.window.start, a_.window.end) == (a[0].min(), a[0].max())
    assert (b_.window.start, b_.window.end) == (0, 5000)
    assert b_.window.min == 0 and b_.window.max == 5000
    # NGFF 0.5 requires a window: None is written as the full data range.
    assert (c_.window.start, c_.window.end) == (a[2].min(), a[2].max())
    assert (c_.window.min, c_.window.max) == (a[2].min(), a[2].max())
    assert d_.window.max == 3


def test_forms_equivalent(tmp_path):
    a = data()
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    p1 = make(tmp_path / "a", a, channels={"x": {"color": "FF0000"}, "y": {}})
    p2 = make(tmp_path / "b", a, channels=[{"label": "x", "color": "FF0000"}, "y"])
    assert channels_of(p1) == channels_of(p2)


def test_omero_object_and_conflict(tmp_path):
    a = data()
    o = Omero(channels=[Channel(label="q"), Channel(label="r")])
    with pytest.warns(DeprecationWarning, match="Omero object in channels"):
        p = make(tmp_path, a, channels=o)
    assert channels_of(p)[0].label == "q"
    with pytest.raises(ValueError, match="not both"):
        Writer(
            tmp_path / "z",
            a,
            list("czyx"),
            UNITS,
            channels=["a", "b"],
            omero_metadata=o,
        )


def test_omero_metadata_deprecated(tmp_path):
    o = Omero(channels=[Channel(label="q"), Channel(label="r")])
    with pytest.warns(DeprecationWarning, match="omero_metadata"):
        Writer(tmp_path / "w", data(), list("czyx"), UNITS, omero_metadata=o)


def test_top_level_omero_deprecated():
    import ome_zarr_io

    with pytest.warns(DeprecationWarning, match="ome_zarr_io.Omero"):
        assert ome_zarr_io.Omero is Omero


def test_no_channels_no_omero(tmp_path):
    p = make(tmp_path, data())
    assert Reader(p)._metadata.ome.omero is None


@pytest.mark.parametrize(
    "kwargs,dims,match",
    [
        ({"channels": ["a"]}, "czyx", "'c' axis has 2"),
        ({"channels": ["a", "b", "c"]}, "czyx", "'c' axis has 2"),
        ({"channels": ["a", "a"]}, "czyx", "duplicate"),
        ({"channels": [{"label": "a", "colour": "x"}, "b"]}, "czyx", "unknown"),
        ({"channels": [{"color": "12"}, "b"]}, "czyx", "6 hex"),
        ({"channels": [{"window": (5, 1)}, "b"]}, "czyx", "start"),
        (
            {"channels": [{"window": Window(start=5, min=0, end=2, max=9)}, "b"]},
            "czyx",
            "min <= start",
        ),
        ({"colors": "random"}, "czyx", "requires channels"),
        ({"channels": ["a"]}, "zyx", "requires a 'c' axis"),
    ],
)
def test_write_time_errors(tmp_path, kwargs, dims, match):
    a = data() if dims == "czyx" else data()[0]
    with pytest.raises(ValueError, match=match):
        Writer(tmp_path / "e.zarr", a, list(dims), UNITS, **kwargs)


def test_fixed_colors_untouched(tmp_path):
    p = make(
        tmp_path, data(3), channels=[{"color": "123456"}, "b", "c"], colors="random"
    )
    ch = channels_of(p)
    assert ch[0].color == "123456"
    assert HEX.match(ch[1].color) and HEX.match(ch[2].color)


def test_get_channel_still_works(tmp_path):
    p = make(tmp_path, data(), channels=["DAPI", "GFP"])
    assert Reader(p).get_channel("GFP", as_type="numpy").shape[0] == 2


# --- validate() semantic checks -----------------------------------------
def _tamper(path, fn):
    import zarr

    root = zarr.open_group(str(path), mode="r+")
    attrs = dict(root.attrs)
    fn(attrs["ome"]["omero"])
    root.attrs.update(attrs)


@pytest.mark.parametrize(
    "mutate,expected",
    [
        (lambda o: o["channels"].pop(), "'c' axis has 2"),
        (lambda o: o["channels"].append(dict(o["channels"][0])), "'c' axis has 2"),
        (
            lambda o: o["channels"][1].update(label=o["channels"][0]["label"]),
            "duplicate",
        ),
        (lambda o: o["channels"][0].update(color="#FF0000"), "6 hex digits"),
        (lambda o: o["channels"][0]["window"].update(start=10**9), "min <= start"),
    ],
)
def test_validate_flags_semantic_channel_problems(tmp_path, mutate, expected):
    p = make(tmp_path, data(), channels={"a": {"color": "FF0000"}, "b": {}})
    assert validate(p).is_valid
    _tamper(p, mutate)
    report = validate(p)
    assert not report.is_valid
    assert any(expected in e.message for e in report.errors)


# --- default colors (NGFF 0.5: every omero channel MUST have a color) ------
def test_channels_without_color_get_one_by_default(tmp_path):
    p = make(tmp_path, data(3), channels=["a", "b", "c"])
    colors = [c.color for c in channels_of(p)]
    assert all(HEX.match(c) for c in colors)
    assert len(set(colors)) == 3
    assert validate(p).is_valid


def test_default_colors_match_colors_random(tmp_path):
    default = make(tmp_path / "d", data(3), channels=["a", "b", "c"])
    explicit = make(tmp_path / "e", data(3), channels=["a", "b", "c"], colors="random")
    assert [c.color for c in channels_of(default)] == [
        c.color for c in channels_of(explicit)
    ]


def test_default_colors_are_reproducible_and_follow_the_seed(tmp_path):
    def colors(name, **kwargs):
        p = make(tmp_path / name, data(3), channels=["a", "b", "c"], **kwargs)
        return [c.color for c in channels_of(p)]

    assert colors("one") == colors("two")
    assert colors("seeded", color_seed=7) != colors("three")


def test_explicit_colors_are_kept_and_avoided(tmp_path):
    p = make(tmp_path, data(3), channels={"a": {"color": "FF0000"}, "b": {}, "c": {}})
    colors = [c.color for c in channels_of(p)]
    assert colors[0] == "FF0000"
    assert all(HEX.match(c) for c in colors[1:])
    assert "FF0000" not in colors[1:] and colors[1] != colors[2]


def test_dict_entries_without_color_get_one(tmp_path):
    p = make(tmp_path, data(2), channels=[{"label": "a"}, {"label": "b", "window": None}])
    assert all(HEX.match(c.color) for c in channels_of(p))


# --- every channel has a window (NGFF 0.5 MUST) ----------------------------
def test_window_none_is_written_as_the_full_range(tmp_path):
    a = data(2)
    p = make(tmp_path, a, channels={"a": {"window": None}, "b": {"window": None}})
    for i, c in enumerate(channels_of(p)):
        assert c.window is not None
        assert (c.window.min, c.window.start) == (a[i].min(), a[i].min())
        assert (c.window.end, c.window.max) == (a[i].max(), a[i].max())
    assert validate(p).is_valid


def test_all_nan_channel_still_gets_a_window(tmp_path):
    a = np.full((2, 4, 16, 16), np.nan, dtype=np.float32)
    a[1] = 1.0
    p = make(tmp_path, a, channels=["empty", "ones"])
    empty, ones = channels_of(p)
    w = empty.window
    assert w is not None and w.min <= w.start <= w.end <= w.max
    assert (ones.window.min, ones.window.max) == (1.0, 1.0)


# --- deprecated Omero objects are completed too ----------------------------
def omero_path(tmp_path, a, omero, dims="czyx", via="omero_metadata", **kw):
    with pytest.warns(DeprecationWarning):
        return make(tmp_path, a, dims, **{via: omero}, **kw)


@pytest.mark.parametrize("via", ["omero_metadata", "channels"])
def test_omero_channels_without_color_or_window_are_completed(tmp_path, via):
    a = data(3)
    omero = Omero(
        channels=[
            Channel(label="full", color="0000FF", window=Window(1, 0, 2, 3)),
            Channel(label="no-window", color="00FF00"),
            Channel(label="bare"),
        ]
    )
    p = omero_path(tmp_path, a, omero, via=via)
    full, no_window, bare = channels_of(p)

    # A complete channel is written exactly as given.
    assert full.color == "0000FF"
    assert (full.window.start, full.window.min, full.window.end, full.window.max) == (
        1,
        0,
        2,
        3,
    )
    # A missing window becomes the channel's full data range.
    assert no_window.color == "00FF00"
    assert (no_window.window.min, no_window.window.max) == (a[1].min(), a[1].max())
    assert (no_window.window.start, no_window.window.end) == (a[1].min(), a[1].max())
    # A missing color becomes a distinct automatic one.
    assert HEX.match(bare.color) and bare.color not in ("0000FF", "00FF00")
    assert (bare.window.min, bare.window.max) == (a[2].min(), a[2].max())
    assert validate(p).is_valid


def test_completing_omero_does_not_modify_the_callers_object(tmp_path):
    omero = Omero(channels=[Channel(label="a"), Channel(label="b")])
    omero_path(tmp_path, data(2), omero)
    assert all(c.color is None and c.window is None for c in omero.channels)


def test_omero_without_a_channel_axis_uses_the_whole_image(tmp_path):
    a = np.arange(4 * 16 * 16, dtype=np.uint16).reshape(4, 16, 16)
    p = omero_path(tmp_path, a, Omero(channels=[Channel(label="only")]), dims="zyx")
    (only,) = channels_of(p)
    assert HEX.match(only.color)
    assert (only.window.min, only.window.max) == (a.min(), a.max())


def test_complete_omero_is_reproducible(tmp_path):
    def colors(name, **kw):
        omero = Omero(channels=[Channel(label="a"), Channel(label="b")])
        p = omero_path(tmp_path / name, data(2), omero, **kw)
        return [c.color for c in channels_of(p)]

    assert colors("one") == colors("two")
    assert colors("seeded", color_seed=5) != colors("three")


def test_omero_colors_are_normalized_and_invalid_ones_rejected(tmp_path):
    omero = Omero(channels=[Channel(label="a", color="#00ff00"), Channel(label="b")])
    p = omero_path(tmp_path / "ok", data(2), omero)
    assert channels_of(p)[0].color == "00FF00"

    bad = Omero(channels=[Channel(label="a", color="green"), Channel(label="b")])
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="6 hex"):
        make(tmp_path / "bad", data(2), omero_metadata=bad)
    assert not (tmp_path / "bad").exists()  # rejected before anything is written
