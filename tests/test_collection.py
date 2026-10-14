"""Step 2: the Measurements record. All tier 1."""

import numpy as np
import pytest

from benchstats import collection


def _random_record(rng, *, n_builds=2, n_invocations=3, n_iterations=5):
    shape = (2, n_builds, n_invocations, n_iterations)
    n = int(np.prod(shape))
    variant, build, invocation, iteration = np.indices(shape).reshape(4, n)
    # A random run order in which iterations increase within each invocation.
    # np.indices lists each invocation's iterations together, in order.
    order = rng.permutation(n).reshape(-1, n_iterations)
    order = np.sort(order, axis=1).ravel()
    return collection.make_measurements(
        rng.lognormal(-7, 0.5, size=n),
        variant=np.where(variant == 0, "A", "B"),
        build=build,
        invocation=invocation,
        iteration=iteration,
        order=order,
        warmup=iteration < 1,
        benchmark="random",
        executions_per_draw=3,
        environment={"host": "test", "cpu": "x86"},
        source="unit test",
    )


def _assert_same(a, b):
    for name in collection.COLUMNS:
        left, right = getattr(a, name), getattr(b, name)
        assert left.dtype == right.dtype, name
        np.testing.assert_array_equal(left, right, err_msg=name)
    assert a.time.tobytes() == b.time.tobytes()  # bit for bit
    for name in ("benchmark", "executions_per_draw", "created", "source"):
        assert getattr(a, name) == getattr(b, name)
    assert dict(a.environment) == dict(b.environment)


# ---------------------------------------------------------------- construction


def test_defaults_give_one_invocation_in_position_order():
    m = collection.make_measurements([3e-3, 1e-3, 2e-3], benchmark="b")
    np.testing.assert_array_equal(m.iteration, [0, 1, 2])
    np.testing.assert_array_equal(m.order, [0, 1, 2])
    assert set(m.variant) == {"A"}
    assert not m.warmup.any()


def test_default_iterations_follow_run_order_within_invocations():
    m = collection.make_measurements([1.0, 2.0, 3.0, 4.0], invocation=[0, 1, 0, 1], order=[3, 2, 1, 0], benchmark="b")
    np.testing.assert_array_equal(m.iteration, [1, 1, 0, 0])


def test_arrays_are_read_only(rng):
    m = _random_record(rng)
    for name in collection.COLUMNS:
        with pytest.raises(ValueError):
            getattr(m, name)[0] = getattr(m, name)[1]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"time": [1.0, 2.0], "build": [0]}, "build: must be 1-D with the same length"),
        ({"time": [1.0, np.nan]}, "time: all values must be finite"),
        ({"time": [1.0, -2.0]}, "time: all values must be positive"),
        ({"time": [1.0, 0.0]}, "time: all values must be positive"),
        ({"time": [1.0, 2.0], "build": [0, -1]}, "build: labels must be non-negative"),
        ({"time": [1.0, 2.0], "invocation": [0.5, 1.0]}, "invocation: labels must be integers"),
        ({"time": [1.0, 2.0], "order": [1, 1]}, "order: values must be unique"),
        ({"time": [1.0, 2.0], "iteration": [0, 2]}, "iteration: within each invocation"),
        ({"time": [1.0, 2.0], "iteration": [1, 0]}, "iteration: within each invocation"),
        ({"time": [[1.0, 2.0]]}, "time: must be 1-D"),
    ],
)
def test_construction_checks_reject_bad_input(kwargs, message):
    time = kwargs.pop("time")
    with pytest.raises(ValueError, match=message):
        collection.make_measurements(time, benchmark="b", **kwargs)


def test_metadata_checks_reject_bad_input():
    with pytest.raises(ValueError, match="benchmark"):
        collection.make_measurements([1.0], benchmark="")
    with pytest.raises(ValueError, match="executions_per_draw"):
        collection.make_measurements([1.0], benchmark="b", executions_per_draw=0)


# ---------------------------------------------------------------- reading and saving


def test_csv_round_trip_is_bit_exact(rng, tmp_path):
    m = _random_record(rng)
    path = tmp_path / "draws.csv"
    collection.save_csv(m, path)
    assert (tmp_path / "draws.json").exists()
    _assert_same(m, collection.load_csv(path))


def test_hdf5_round_trip_is_bit_exact(rng, tmp_path):
    pytest.importorskip("tables")
    m = _random_record(rng)
    path = tmp_path / "draws.h5"
    collection.save_hdf5(m, path)
    _assert_same(m, collection.load_hdf5(path))


def test_hdf5_file_opens_without_pandas(rng, tmp_path):
    pytest.importorskip("tables")
    h5py = pytest.importorskip("h5py")
    import json

    m = _random_record(rng)
    path = tmp_path / "draws.h5"
    collection.save_hdf5(m, path, key="draws")
    with h5py.File(path, "r") as f:
        table = f["draws/table"][...]
        metadata = json.loads(f["draws"].attrs["benchstats_metadata"])
    np.testing.assert_array_equal(table["time"], m.time)
    np.testing.assert_array_equal(table["order"], m.order)
    np.testing.assert_array_equal(table["variant"].astype(str), m.variant)
    assert metadata["benchmark"] == "random"


def test_read_table_maps_columns_and_units(tmp_path):
    path = tmp_path / "external.csv"
    path.write_text("ns,run,iter,commit\n1500,0,0,abc\n2500,0,1,abc\n1700,1,0,def\n")
    m = collection.read_table(
        path,
        {"ns": "time", "run": "invocation", "iter": "iteration", "commit": "variant"},
        benchmark="ext",
        time_unit="ns",
    )
    np.testing.assert_allclose(m.time, [1.5e-6, 2.5e-6, 1.7e-6])
    np.testing.assert_array_equal(m.variant, ["abc", "abc", "def"])
    np.testing.assert_array_equal(m.invocation, [0, 0, 1])
    assert m.source == str(path)


def test_read_table_reports_missing_columns(tmp_path):
    path = tmp_path / "external.csv"
    path.write_text("t\n1\n")
    with pytest.raises(ValueError, match="no columns"):
        collection.read_table(path, {"seconds": "time"}, benchmark="ext")
    with pytest.raises(ValueError, match="unknown target"):
        collection.read_table(path, {"t": "duration"}, benchmark="ext")


# ---------------------------------------------------------------- arrays for estimators


def test_flatten_respects_run_order_and_warmup(rng):
    m = _random_record(rng)
    for variant in ("A", "B"):
        mask = (m.variant == variant) & ~m.warmup
        expected = m.time[mask][np.argsort(m.order[mask])]
        np.testing.assert_array_equal(collection.flatten(m, variant=variant), expected)
        with_warmup = collection.flatten(m, variant=variant, include_warmup=True)
        assert with_warmup.size == (m.variant == variant).sum()
        assert np.all(np.diff(collection.run_order(m, variant=variant, include_warmup=True)) > 0)


def test_flatten_needs_variant_when_several(rng):
    m = _random_record(rng)
    with pytest.raises(ValueError, match="several variants"):
        collection.flatten(m)
    with pytest.raises(ValueError, match="not in the record"):
        collection.flatten(m, variant="C")


@pytest.mark.parametrize("level", ["build", "invocation"])
def test_group_labels_line_up_with_flatten(rng, level):
    m = _random_record(rng)
    times = collection.flatten(m, variant="B")
    labels = collection.group_labels(m, level, variant="B")
    assert labels.shape == times.shape
    for label in np.unique(labels):
        rows = np.isin(m.time, times[labels == label])
        keys = {(b, i) for b, i in zip(m.build[rows], m.invocation[rows])}
        builds = {b for b, _ in keys}
        assert len(builds if level == "build" else keys) == 1
    expected_groups = 2 if level == "build" else 6
    assert np.unique(labels).size == expected_groups


def test_mark_warmup_returns_new_record(rng):
    m = _random_record(rng)
    before = m.warmup.copy()
    marked = collection.mark_warmup(m, 2)
    np.testing.assert_array_equal(m.warmup, before)
    np.testing.assert_array_equal(marked.warmup, m.iteration < 2)
    per_invocation = collection.mark_warmup(m, {("A", 0, 0): 4})
    assert per_invocation.warmup.sum() == 4
    with pytest.raises(ValueError, match="no invocation"):
        collection.mark_warmup(m, {("C", 0, 0): 1})
