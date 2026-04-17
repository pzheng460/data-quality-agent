"""Tests for the compute backend abstraction."""

from __future__ import annotations

import pytest

from dq.runner.backend import ComputeBackend, LocalBackend, make_backend


def _square(x: int) -> int:
    return x * x


def _double(x: int) -> int:
    return x * 2


def test_make_backend_local_default():
    b = make_backend()
    assert b.name == "local"
    assert isinstance(b, LocalBackend)


def test_make_backend_unknown_raises():
    with pytest.raises(ValueError):
        make_backend("nope")


def test_local_backend_cpu_map_preserves_all_items():
    b = LocalBackend(cpu_workers=2)
    result = sorted(b.map(_square, [1, 2, 3, 4], kind="cpu"))
    assert result == [1, 4, 9, 16]


def test_local_backend_io_map_preserves_order():
    b = LocalBackend(io_workers=4)
    # ThreadPoolExecutor.map preserves order
    result = list(b.map(_double, [1, 2, 3], kind="io"))
    assert result == [2, 4, 6]


def test_local_backend_cpu_single_worker_runs_inline():
    # cpu_workers=1 → runs in main process, no multiprocessing overhead
    b = LocalBackend(cpu_workers=1)
    result = list(b.map(_square, [1, 2, 3], kind="cpu"))
    assert result == [1, 4, 9]


def test_local_backend_rejects_unknown_kind():
    b = LocalBackend(cpu_workers=1)
    with pytest.raises(ValueError):
        list(b.map(_double, [1, 2], kind="gpu"))
