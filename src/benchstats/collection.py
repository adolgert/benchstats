"""Collection: the Measurements record.

One record type holds every draw with its labels, in long format (one row per
draw). Everything else here is a function: build a record, read and save it,
and hand estimators plain arrays. ``flatten`` is the only place the hierarchy
is collapsed.

This module imports nothing from the rest of the package.
"""

import csv
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

import numpy as np

LABELS = ("variant", "build", "invocation", "iteration", "order", "warmup")
COLUMNS = ("time", *LABELS)
LEVELS = ("build", "invocation")
TIME_UNITS = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
_INTEGER_LABELS = ("build", "invocation", "iteration", "order")


@dataclass(frozen=True)
class Measurements:
    """Every draw of a benchmark experiment with its labels.

    Columns, one entry per draw:

    - time: float64 seconds for the draw.
    - variant: which version ran, such as "A", "B" or a commit id.
    - build: which build of that variant.
    - invocation: which process run within the build.
    - iteration: position within the invocation, from 0.
    - order: position in the run order of the whole experiment.
    - warmup: whether the draw belongs to the warmup.

    Metadata: benchmark name, executions per draw, environment, creation
    time (ISO 8601) and source. Build records with ``make_measurements``.
    The arrays are read-only.
    """

    time: np.ndarray
    variant: np.ndarray
    build: np.ndarray
    invocation: np.ndarray
    iteration: np.ndarray
    order: np.ndarray
    warmup: np.ndarray
    benchmark: str
    executions_per_draw: int = 1
    environment: Mapping[str, str] = field(default_factory=dict)
    created: str = ""
    source: str = ""

    def __post_init__(self):
        _check_record(self)


# ---------------------------------------------------------------- checks


def _check_record(m):
    n = m.time.shape[0] if m.time.ndim == 1 else -1
    if m.time.ndim != 1:
        raise ValueError("time: must be 1-D")
    for name in LABELS:
        column = getattr(m, name)
        if column.ndim != 1 or column.shape[0] != n:
            raise ValueError(f"{name}: must be 1-D with the same length as time ({n})")
    if m.time.dtype != np.float64:
        raise ValueError("time: must be float64 seconds")
    if not np.all(np.isfinite(m.time)):
        raise ValueError("time: all values must be finite")
    if not np.all(m.time > 0):
        raise ValueError("time: all values must be positive")
    for name in _INTEGER_LABELS:
        column = getattr(m, name)
        if column.dtype.kind not in "iu":
            raise ValueError(f"{name}: labels must be integers")
        if np.any(column < 0):
            raise ValueError(f"{name}: labels must be non-negative")
    if m.warmup.dtype != np.bool_:
        raise ValueError("warmup: must be boolean")
    if m.variant.dtype.kind != "U":
        raise ValueError("variant: must be strings")
    if np.unique(m.order).size != n:
        raise ValueError("order: values must be unique")
    _check_iterations(m)
    if not isinstance(m.benchmark, str) or not m.benchmark:
        raise ValueError("benchmark: must be a non-empty string")
    if not isinstance(m.executions_per_draw, int) or m.executions_per_draw < 1:
        raise ValueError("executions_per_draw: must be a positive integer")


def _invocation_codes(variant, build, invocation):
    """One integer per distinct (variant, build, invocation), sorted."""
    _, variant_code = np.unique(variant, return_inverse=True)
    keys = np.stack([variant_code, build, invocation], axis=1)
    _, codes = np.unique(keys, axis=0, return_inverse=True)
    return codes.ravel()


def _check_iterations(m):
    """Within each invocation, iterations run 0, 1, 2, ... in run order."""
    if m.time.size == 0:
        return
    codes = _invocation_codes(m.variant, m.build, m.invocation)
    idx = np.lexsort((m.order, codes))
    codes, iteration = codes[idx], m.iteration[idx]
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    position = np.arange(codes.size) - np.repeat(starts, np.diff(np.r_[starts, codes.size]))
    if not np.array_equal(iteration, position):
        raise ValueError("iteration: within each invocation, iterations must run 0, 1, 2, ... in run order")


def _frozen(array, dtype):
    out = np.array(array, dtype=dtype, copy=True)
    out.setflags(write=False)
    return out


def _iteration_in_order(codes, order):
    """Position of each draw within its invocation, by run order."""
    idx = np.lexsort((order, codes))
    sorted_codes = codes[idx]
    starts = np.flatnonzero(np.r_[True, sorted_codes[1:] != sorted_codes[:-1]])
    position = np.arange(codes.size) - np.repeat(starts, np.diff(np.r_[starts, codes.size]))
    out = np.empty_like(position)
    out[idx] = position
    return out


# ---------------------------------------------------------------- building


def make_measurements(
    time,
    *,
    variant=None,
    build=None,
    invocation=None,
    iteration=None,
    order=None,
    warmup=None,
    benchmark,
    executions_per_draw=1,
    environment=None,
    created=None,
    source="",
):
    """Build a Measurements record from raw times in seconds.

    Missing labels default to one variant ("A"), one build and one
    invocation. ``order`` defaults to position. ``iteration`` defaults to the
    position of each draw within its invocation, in run order. ``variant`` may
    be one string for all draws. Raises ValueError naming the failed check.
    """
    time = np.asarray(time, dtype=np.float64)
    n = time.shape[0] if time.ndim == 1 else 0
    if time.ndim != 1:
        raise ValueError("time: must be 1-D")

    if variant is None or isinstance(variant, str):
        variant = np.full(n, variant or "A", dtype=str)
    variant = np.asarray(variant, dtype=str)
    build = _as_int_labels("build", build, n)
    invocation = _as_int_labels("invocation", invocation, n)
    order = _as_int_labels("order", order, n, default=np.arange(n))
    if iteration is None and variant.shape == build.shape == invocation.shape == order.shape == (n,):
        iteration = _iteration_in_order(_invocation_codes(variant, build, invocation), order)
    iteration = _as_int_labels("iteration", iteration, n)
    warmup = np.zeros(n, dtype=bool) if warmup is None else np.asarray(warmup, dtype=bool)
    return Measurements(
        time=_frozen(time, np.float64),
        variant=_frozen(variant, str),
        build=_frozen(build, np.int64),
        invocation=_frozen(invocation, np.int64),
        iteration=_frozen(iteration, np.int64),
        order=_frozen(order, np.int64),
        warmup=_frozen(warmup, bool),
        benchmark=benchmark,
        executions_per_draw=executions_per_draw,
        environment=MappingProxyType({str(k): str(v) for k, v in (environment or {}).items()}),
        created=created or datetime.now(UTC).isoformat(timespec="seconds"),
        source=source,
    )


def _as_int_labels(name, value, n, default=None):
    if value is None:
        return np.zeros(n, dtype=np.int64) if default is None else np.asarray(default, dtype=np.int64)
    array = np.asarray(value)
    if array.dtype.kind == "f":
        if not np.all(np.isfinite(array)) or not np.all(array == np.round(array)):
            raise ValueError(f"{name}: labels must be integers")
        array = array.astype(np.int64)
    if array.dtype.kind not in "iub":
        raise ValueError(f"{name}: labels must be integers")
    return array.astype(np.int64)


def _replace(m, **changes):
    fields = {name: getattr(m, name) for name in m.__dataclass_fields__}
    fields.update(changes)
    return Measurements(**fields)


def mark_warmup(m, counts):
    """Return a new record with the first ``counts`` iterations of each invocation as warmup.

    ``counts`` is one integer for all invocations, or a mapping from
    (variant, build, invocation) to an integer, such as warmup_mser results.
    Invocations missing from the mapping get no warmup. Existing warmup
    marks are replaced. The original record is unchanged.
    """
    if isinstance(counts, Mapping):
        per_draw = np.zeros(m.time.size, dtype=np.int64)
        for (variant, build, invocation), count in counts.items():
            mask = (m.variant == variant) & (m.build == build) & (m.invocation == invocation)
            if not mask.any():
                raise ValueError(f"counts: no invocation {(variant, build, invocation)!r} in the record")
            per_draw[mask] = int(count)
    else:
        per_draw = np.full(m.time.size, int(counts), dtype=np.int64)
    if np.any(per_draw < 0):
        raise ValueError("counts: must be non-negative")
    return _replace(m, warmup=_frozen(m.iteration < per_draw, bool))


# ---------------------------------------------------------------- arrays for estimators


def _selection(m, variant, include_warmup):
    """Indices of the selected draws, in run order."""
    variants = np.unique(m.variant)
    if variant is None:
        if variants.size != 1:
            raise ValueError(f"variant: the record holds several variants {list(variants)}; name one")
        mask = np.ones(m.time.size, dtype=bool)
    else:
        if variant not in variants:
            raise ValueError(f"variant: {variant!r} is not in the record")
        mask = m.variant == variant
    if not include_warmup:
        mask &= ~m.warmup
    idx = np.flatnonzero(mask)
    return idx[np.argsort(m.order[idx], kind="stable")]


def flatten(m, *, variant=None, include_warmup=False):
    """Times of one variant as a 1-D array in run order (raw seconds).

    This is the only place the hierarchy is collapsed. ``variant`` may be
    omitted when the record holds one variant. Warmup draws are left out
    unless ``include_warmup``.
    """
    return m.time[_selection(m, variant, include_warmup)].copy()


def run_order(m, *, variant=None, include_warmup=False):
    """Run-order positions aligned with ``flatten`` for the same arguments."""
    return m.order[_selection(m, variant, include_warmup)].copy()


def group_labels(m, level, *, variant=None, include_warmup=False):
    """Integer labels 0, 1, ... for builds or invocations, aligned with ``flatten``.

    ``level`` is "build" or "invocation". Invocation labels distinguish
    invocations of different builds.
    """
    if level not in LEVELS:
        raise ValueError(f"level: must be one of {LEVELS}")
    idx = _selection(m, variant, include_warmup)
    if level == "build":
        keys = np.stack([m.build[idx]], axis=1)
    else:
        keys = np.stack([m.build[idx], m.invocation[idx]], axis=1)
    _, labels = np.unique(keys, axis=0, return_inverse=True)
    return labels.ravel().astype(np.int64)


# ---------------------------------------------------------------- reading and saving


def _metadata(m):
    return {
        "benchmark": m.benchmark,
        "executions_per_draw": m.executions_per_draw,
        "environment": dict(m.environment),
        "created": m.created,
        "source": m.source,
    }


def _from_columns(columns, metadata):
    return make_measurements(
        columns["time"],
        variant=columns["variant"],
        build=columns["build"],
        invocation=columns["invocation"],
        iteration=columns["iteration"],
        order=columns["order"],
        warmup=columns["warmup"],
        **metadata,
    )


def _parse_bool(text):
    lowered = text.strip().lower()
    if lowered in ("true", "1", "yes"):
        return True
    if lowered in ("false", "0", "no", ""):
        return False
    raise ValueError(f"warmup: cannot read {text!r} as a boolean")


def read_table(path, columns, *, benchmark, time_unit="s", executions_per_draw=1, environment=None):
    """Read timings produced elsewhere (Fortran, Julia, GPU code) from a CSV file.

    ``columns`` maps the file's column names to ours: "time" is required, and
    any of "variant", "build", "invocation", "iteration", "order" and
    "warmup" may be given; missing labels get the defaults of
    ``make_measurements``. ``time_unit`` is one of "s", "ms", "us", "ns".
    """
    if time_unit not in TIME_UNITS:
        raise ValueError(f"time_unit: must be one of {tuple(TIME_UNITS)}")
    unknown = set(columns.values()) - set(COLUMNS)
    if unknown:
        raise ValueError(f"columns: unknown target columns {sorted(unknown)}")
    if "time" not in columns.values():
        raise ValueError("columns: must map one file column to 'time'")
    raw = {ours: [] for ours in columns.values()}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = set(columns) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"columns: file has no columns {sorted(missing)}")
        for row in reader:
            for theirs, ours in columns.items():
                raw[ours].append(row[theirs])
    parsed = {"time": np.array([float(v) for v in raw["time"]]) * TIME_UNITS[time_unit]}
    for name, values in raw.items():
        if name == "variant":
            parsed[name] = np.array(values, dtype=str)
        elif name == "warmup":
            parsed[name] = np.array([_parse_bool(v) for v in values], dtype=bool)
        elif name != "time":
            parsed[name] = np.array([int(v) for v in values], dtype=np.int64)
    return make_measurements(
        parsed.pop("time"),
        benchmark=benchmark,
        executions_per_draw=executions_per_draw,
        environment=environment,
        source=str(path),
        **parsed,
    )


def _metadata_path(path):
    return Path(path).with_suffix(".json")


def save_csv(m, path):
    """Save draws as CSV (one row per draw) and metadata as JSON beside it.

    The JSON file has the same name with the suffix ".json". Times are
    written with 17 significant digits so they reload bit for bit.
    """
    path = Path(path)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for row in zip(m.time, m.variant, m.build, m.invocation, m.iteration, m.order, m.warmup):
            t, variant, build, invocation, iteration, order, warmup = row
            writer.writerow([format(t, ".17g"), variant, build, invocation, iteration, order, str(bool(warmup)).lower()])
    _metadata_path(path).write_text(json.dumps(_metadata(m), indent=2) + "\n")


def load_csv(path):
    """Load a record written by ``save_csv``."""
    path = Path(path)
    metadata = json.loads(_metadata_path(path).read_text())
    columns = {name: [] for name in COLUMNS}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError(f"{path}: expected columns {COLUMNS}")
        for row in reader:
            for name in COLUMNS:
                columns[name].append(row[name])
    parsed = {
        "time": np.array([float(v) for v in columns["time"]]),
        "variant": np.array(columns["variant"], dtype=str),
        "warmup": np.array([_parse_bool(v) for v in columns["warmup"]], dtype=bool),
    }
    for name in _INTEGER_LABELS:
        parsed[name] = np.array([int(v) for v in columns[name]], dtype=np.int64)
    return _from_columns(parsed, metadata)


def _require_pandas():
    try:
        import pandas
        import tables  # noqa: F401  (pandas needs PyTables for HDF5)
    except ImportError as error:
        raise ImportError("HDF5 support needs pandas and PyTables: pip install 'benchstats[hdf5]'") from error
    return pandas


def save_hdf5(m, path, *, key="draws"):
    """Save draws as a pandas DataFrame in an HDF5 file, with metadata as a JSON attribute.

    Uses pandas' table format with every column as its own field, so other
    HDF5 readers can open the file. Needs ``benchstats[hdf5]``.
    """
    pd = _require_pandas()
    frame = pd.DataFrame({name: np.asarray(getattr(m, name)) for name in COLUMNS})
    frame["variant"] = frame["variant"].astype(object)
    with pd.HDFStore(path, mode="a") as store:
        store.put(key, frame, format="table", data_columns=True)
        store.get_storer(key).attrs.benchstats_metadata = json.dumps(_metadata(m))


def load_hdf5(path, *, key="draws"):
    """Load a record written by ``save_hdf5``. Needs ``benchstats[hdf5]``."""
    pd = _require_pandas()
    with pd.HDFStore(path, mode="r") as store:
        frame = store.get(key)
        metadata = json.loads(store.get_storer(key).attrs.benchstats_metadata)
    columns = {name: frame[name].to_numpy() for name in COLUMNS}
    columns["variant"] = columns["variant"].astype(str)
    return _from_columns(columns, metadata)


def describe(m):
    """A one-line summary of the record's size and labels, for printing."""
    n_inv = np.unique(_invocation_codes(m.variant, m.build, m.invocation)).size if m.time.size else 0
    variants = ", ".join(np.unique(m.variant))
    return (
        f"{m.benchmark}: {m.time.size} draws, variants [{variants}], {n_inv} invocations, "
        f"{int(m.warmup.sum())} warmup, {m.executions_per_draw} execution(s) per draw"
    )
