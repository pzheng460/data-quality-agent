"""Compute backend abstraction.

Decouples pipeline stages from the concrete parallel-execution primitive.

Available backends:
  - local  (default): multiprocessing for CPU work, threads for IO work
  - dask:             distributed.Client (requires `pip install 'dask[distributed]'`)
  - ray:              ray.remote (requires `pip install ray`)

CPU vs IO distinction:
  - kind="cpu": CPU-bound work (filters, LaTeXML). Uses processes locally.
  - kind="io":  IO-bound work (LLM API calls, HTTP downloads). Uses threads.

For distributed backends (dask, ray) the kind is advisory; the scheduler
picks workers either way.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Iterator, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class ComputeBackend(ABC):
    """Parallel compute backend. Implementations: Local / Dask / Ray."""

    name: str = "abstract"

    @abstractmethod
    def map(
        self,
        fn: Callable[[T], R],
        items: Iterable[T],
        *,
        kind: str = "cpu",
        chunksize: int = 1,
    ) -> Iterator[R]:
        """Apply fn to each item in parallel. Yields results in arbitrary order."""

    def shutdown(self) -> None:
        """Release long-lived resources. No-op by default."""
        return None


def _default_cpu_workers() -> int:
    return max(1, min((os.cpu_count() or 1) // 4, 32))


class LocalBackend(ComputeBackend):
    """Single-machine backend. Default choice — no external deps.

    Uses multiprocessing for CPU work (spawn context) and threads for IO.
    """

    name = "local"

    def __init__(
        self,
        cpu_workers: int | None = None,
        io_workers: int | None = None,
        **_ignored: object,
    ):
        self.cpu_workers = int(cpu_workers) if cpu_workers else _default_cpu_workers()
        self.io_workers = int(io_workers) if io_workers else 8

    def map(self, fn, items, *, kind="cpu", chunksize=1):
        if kind == "cpu":
            if self.cpu_workers <= 1:
                for x in items:
                    yield fn(x)
                return
            from multiprocessing import get_context
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            ctx = get_context("spawn")
            with ctx.Pool(self.cpu_workers) as pool:
                yield from pool.imap_unordered(fn, items, chunksize=chunksize)

        elif kind == "io":
            if self.io_workers <= 1:
                for x in items:
                    yield fn(x)
                return
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.io_workers) as ex:
                yield from ex.map(fn, items)

        else:
            raise ValueError(f"Unknown kind: {kind!r}")


class DaskBackend(ComputeBackend):
    """Dask distributed backend.

    Args:
        scheduler_address: Connect to an existing Dask scheduler (e.g. "tcp://...").
        n_workers: If no address, start a LocalCluster with N workers.
    """

    name = "dask"

    def __init__(
        self,
        scheduler_address: str | None = None,
        n_workers: int = 4,
        **_ignored: object,
    ) -> None:
        try:
            from dask.distributed import Client, LocalCluster  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "DaskBackend needs dask[distributed]. "
                "Install with: pip install 'dask[distributed]'"
            ) from e

        if scheduler_address:
            self._client = Client(scheduler_address)
            self._cluster = None
            logger.info("Connected to Dask scheduler: %s", scheduler_address)
        else:
            self._cluster = LocalCluster(n_workers=n_workers, processes=True)
            self._client = Client(self._cluster)
            logger.info("Started Dask LocalCluster with %d workers", n_workers)

    def map(self, fn, items, *, kind="cpu", chunksize=1):
        # Dask scatters and schedules. `kind` is only a hint.
        items_list = list(items)
        if not items_list:
            return
        from dask.distributed import as_completed  # type: ignore
        futures = self._client.map(fn, items_list)
        for fut in as_completed(futures):
            yield fut.result()

    def shutdown(self):
        try:
            self._client.close()
        except Exception as e:  # pragma: no cover
            logger.debug("Dask client close failed: %s", e)
        if self._cluster is not None:
            try:
                self._cluster.close()
            except Exception as e:  # pragma: no cover
                logger.debug("Dask cluster close failed: %s", e)


class RayBackend(ComputeBackend):
    """Ray distributed backend.

    Args:
        address: Ray scheduler address (None = local Ray cluster)
    """

    name = "ray"

    def __init__(self, address: str | None = None, **_ignored: object) -> None:
        try:
            import ray  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "RayBackend needs the 'ray' package. Install with: pip install ray"
            ) from e
        if not ray.is_initialized():
            if address:
                ray.init(address=address)
            else:
                ray.init(ignore_reinit_error=True)

    def map(self, fn, items, *, kind="cpu", chunksize=1):
        import ray  # type: ignore

        items_list = list(items)
        if not items_list:
            return
        remote_fn = ray.remote(fn)
        pending = [remote_fn.remote(x) for x in items_list]
        while pending:
            done, pending = ray.wait(pending, num_returns=1)
            yield ray.get(done[0])

    def shutdown(self):
        # Keep the Ray cluster alive for subsequent calls.
        pass


def make_backend(name: str | None = "local", **kwargs: Any) -> ComputeBackend:
    """Instantiate a backend by name. Used by PhaseEngine."""
    name = (name or "local").lower()
    if name == "local":
        return LocalBackend(**kwargs)
    if name == "dask":
        return DaskBackend(**kwargs)
    if name == "ray":
        return RayBackend(**kwargs)
    raise ValueError(f"Unknown backend: {name!r} (options: local, dask, ray)")
