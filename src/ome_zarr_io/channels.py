"""Plain-Python channel specifications resolved into OMERO display metadata."""

import colorsys
import random
import re
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .schema_models import Channel, Omero, Window

_HEX_COLOR = re.compile(r"^[0-9A-Fa-f]{6}$")
_GOLDEN = 0.6180339887498949
_CHANNEL_KEYS = {"label", "color", "window", "family", "active"}
_MANY_CHANNELS = 12
_MIN_HUE_DISTANCE = 0.05


def normalize_color(color: str) -> str:
    """Return ``color`` as uppercase ``RRGGBB`` (a leading "#" is stripped)."""
    if not isinstance(color, str):
        raise ValueError(f"channel color must be a hex string, got {color!r}")
    stripped = color.lstrip("#") if color.startswith("#") else color
    if not _HEX_COLOR.match(stripped):
        raise ValueError(f"channel color must be 6 hex digits, got {color!r}")
    return stripped.upper()


def _hue(color: str) -> float:
    r, g, b = (int(color[i : i + 2], 16) / 255 for i in (0, 2, 4))
    return colorsys.rgb_to_hsv(r, g, b)[0]


def _hue_distance(a: float, b: float) -> float:
    d = abs(a - b) % 1.0
    return min(d, 1.0 - d)


def random_colors(n: int, seed: int = 0, avoid: Sequence[str] = ()) -> List[str]:
    """Generate ``n`` visually distinct ``RRGGBB`` colors.

    Hues are spaced by the golden ratio, so color *i* depends only on ``i`` and
    ``seed``: asking for more colors never changes the earlier ones. Hues close to
    a color in ``avoid`` are skipped. A different ``seed`` gives a different palette.
    """
    h0 = 0.0 if seed == 0 else random.Random(seed).random()
    avoid_hues = [_hue(normalize_color(c)) for c in avoid]
    if n > _MANY_CHANNELS:
        warnings.warn(
            f"{n} colors requested; more than {_MANY_CHANNELS} are hard to tell apart",
            UserWarning,
            stacklevel=2,
        )
    colors = []
    step = 0
    for _ in range(n):
        for _attempt in range(50):
            hue = (h0 + step * _GOLDEN) % 1.0
            step += 1
            if all(_hue_distance(hue, a) >= _MIN_HUE_DISTANCE for a in avoid_hues):
                break
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
        colors.append(f"{round(r * 255):02X}{round(g * 255):02X}{round(b * 255):02X}")
    return colors


def auto_contrast(values: np.ndarray) -> Optional[Tuple[float, float]]:
    """ImageJ/Fiji "Auto" display range for one channel, or ``None`` if no finite data."""
    values = np.asarray(values).ravel()
    if values.dtype.kind == "f":
        values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    lo, hi = values.min(), values.max()
    if hi <= lo:
        return float(lo), float(hi)
    hist, _ = np.histogram(values, bins=256, range=(float(lo), float(hi)))
    n = values.size
    limit, threshold = n // 10, n // 5000
    counts = np.where(hist > limit, 0, hist)
    hits = np.nonzero(counts > threshold)[0]
    if hits.size == 0 or hits[-1] <= hits[0]:
        return float(lo), float(hi)
    bin_size = (float(hi) - float(lo)) / 256
    return float(lo) + hits[0] * bin_size, float(lo) + hits[-1] * bin_size


@dataclass
class ChannelSpec:
    """One user-supplied channel, before data statistics are known."""

    label: Optional[str] = None
    color: Optional[str] = None
    window: Any = "auto"
    family: Optional[str] = None
    active: Optional[bool] = None
    auto_color: bool = False


def _spec_from_dict(d: Dict[str, Any], label: Optional[str]) -> ChannelSpec:
    unknown = set(d) - _CHANNEL_KEYS
    if unknown:
        raise ValueError(
            f"unknown channel key(s) {sorted(unknown)}; allowed: {sorted(_CHANNEL_KEYS)}"
        )
    color = d.get("color")
    auto_color = isinstance(color, str) and color.lower() == "random"
    if color is not None and not auto_color:
        color = normalize_color(color)
    return ChannelSpec(
        label=d.get("label", label),
        color=None if auto_color else color,
        window=d.get("window", "auto"),
        family=d.get("family"),
        active=d.get("active"),
        auto_color=auto_color,
    )


def _specs_from_mapping(mapping: Dict[str, Any]) -> List[ChannelSpec]:
    specs = []
    for label, options in mapping.items():
        if options is not None and not isinstance(options, dict):
            raise ValueError(f"channel {label!r} must map to a dict, got {options!r}")
        specs.append(_spec_from_dict(options or {}, str(label)))
    return specs


def _specs_from_sequence(items: Sequence[Any]) -> List[ChannelSpec]:
    specs = []
    for item in items:
        if isinstance(item, str):
            specs.append(ChannelSpec(label=item))
        elif isinstance(item, dict):
            specs.append(_spec_from_dict(item, None))
        else:
            raise ValueError(f"unsupported channel entry {item!r}")
    return specs


def _check_unique_labels(specs: List[ChannelSpec]) -> None:
    labels = [s.label for s in specs if s.label is not None]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(f"duplicate channel labels: {duplicates}")


def _check_explicit_window(spec: ChannelSpec, index: int) -> None:
    """Validate windows that are fully known before the data is read."""
    name = f"channel {spec.label if spec.label is not None else index}"
    window = spec.window
    if isinstance(window, Window):
        _check_window(window, name)
    elif (
        isinstance(window, (tuple, list)) and len(window) == 2 and window[0] > window[1]
    ):
        raise ValueError(
            f"{name}: window start {window[0]} is greater than end {window[1]}"
        )


def parse_channels(
    spec: Union[Dict[str, Any], Sequence[Any], Omero],
    colors: Optional[str] = None,
) -> Union[List[ChannelSpec], Omero]:
    """Normalize the accepted ``channels=`` forms; an ``Omero`` object passes through."""
    if colors not in (None, "random"):
        raise ValueError(f"colors must be None or 'random', got {colors!r}")
    if isinstance(spec, Omero):
        return spec

    if isinstance(spec, dict):
        specs = _specs_from_mapping(spec)
    elif isinstance(spec, (list, tuple)):
        specs = _specs_from_sequence(spec)
    else:
        raise ValueError(f"unsupported channels specification: {type(spec).__name__}")

    _check_unique_labels(specs)
    for index, channel in enumerate(specs):
        _check_explicit_window(channel, index)
        if colors == "random" and channel.color is None:
            channel.auto_color = True
    return specs


def _needs_stats(spec: ChannelSpec) -> bool:
    w = spec.window
    if w is None:
        return False
    if isinstance(w, Window):
        return False
    if isinstance(w, dict):
        return not {"start", "end", "min", "max"} <= set(w)
    return True


def _check_window(w: Window, where: str) -> Window:
    if not (w.min <= w.start <= w.end <= w.max):
        raise ValueError(
            f"{where}: window must satisfy min <= start <= end <= max, got "
            f"min={w.min}, start={w.start}, end={w.end}, max={w.max}"
        )
    return w


def _num(x: Any, integer: bool) -> Union[int, float]:
    return int(x) if integer else float(x)


def _resolve_window(
    spec: ChannelSpec, values: Optional[np.ndarray], where: str
) -> Optional[Window]:
    w = spec.window
    if w is None:
        return None
    if isinstance(w, Window):
        return _check_window(w, where)
    if isinstance(w, dict):
        if {"start", "end", "min", "max"} <= set(w):
            return _check_window(Window(**{k: w[k] for k in w}), where)
        raise ValueError(f"{where}: window dict needs start, end, min and max")

    assert values is not None
    finite = values[np.isfinite(values)] if values.dtype.kind == "f" else values
    if finite.size == 0:
        return None
    integer = finite.dtype.kind in "iu"
    dmin, dmax = _num(finite.min(), integer), _num(finite.max(), integer)

    if w == "minmax":
        return Window(start=dmin, min=dmin, end=dmax, max=dmax)
    if w == "auto":
        rng = auto_contrast(finite)
        assert rng is not None
        start, end = rng
        return Window(start=start, min=dmin, end=end, max=dmax)
    if isinstance(w, (tuple, list)) and len(w) == 2:
        start, end = w
        if start > end:
            raise ValueError(f"{where}: window start {start} is greater than end {end}")
        return Window(start=start, min=min(dmin, start), end=end, max=max(dmax, end))
    raise ValueError(
        f"{where}: window must be 'auto', 'minmax', (start, end), Window, dict or None"
        f", got {w!r}"
    )


def resolve_channels(
    specs: List[ChannelSpec],
    level0: Optional[np.ndarray],
    c_axis: int,
    color_seed: int = 0,
) -> Omero:
    """Turn normalized channel specs into an ``Omero`` object using the level-0 data."""
    n = len(specs)
    fixed = [s.color for s in specs if s.color]
    auto_pool = (
        random_colors(n, color_seed, avoid=fixed)
        if any(s.auto_color for s in specs)
        else []
    )

    channels = []
    for i, s in enumerate(specs):
        where = f"channel {s.label if s.label is not None else i}"
        values = None
        if _needs_stats(s):
            if level0 is None:
                raise ValueError("image data is required to compute channel windows")
            values = np.take(level0, i, axis=c_axis)
        color = auto_pool[i] if s.auto_color else s.color
        channels.append(
            Channel(
                window=_resolve_window(s, values, where),
                label=s.label,
                family=s.family,
                color=color,
                active=s.active,
            )
        )
    return Omero(channels=channels)


def any_needs_stats(specs: List[ChannelSpec]) -> bool:
    return any(_needs_stats(s) for s in specs)
