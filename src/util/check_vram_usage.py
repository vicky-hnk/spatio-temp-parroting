import time
import contextlib
import torch

def _fmt_mb(b):
    return f"{b/1024/1024:.2f} MiB"

@contextlib.contextmanager
def track_cuda_memory(tag="", device=None, verbose=True):
    """Print CUDA memory stats (delta/peak) for the wrapped block."""
    if not torch.cuda.is_available():
        t0 = time.perf_counter()
        yield
        t1 = time.perf_counter()
        if verbose:
            print(f"[mem:{tag}] CUDA not available (CPU only). Elapsed {t1 - t0:.3f}s")
        return

    dev = torch.device("cuda", device) if device is not None else torch.device("cuda")
    torch.cuda.synchronize(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    a0 = torch.cuda.memory_allocated(dev)
    r0 = torch.cuda.memory_reserved(dev)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        torch.cuda.synchronize(dev)
        t1 = time.perf_counter()
        a1 = torch.cuda.memory_allocated(dev)
        r1 = torch.cuda.memory_reserved(dev)
        peak = torch.cuda.max_memory_allocated(dev)  # peak allocated during the block

        if verbose:
            print(
                f"[mem:{tag}] "
                f"alloc Δ {_fmt_mb(a1 - a0)} (→ {_fmt_mb(a1)}), "
                f"reserved Δ {_fmt_mb(r1 - r0)} (→ {_fmt_mb(r1)}), "
                f"peak in block {_fmt_mb(max(0, peak - a0))}, "
                f"time {t1 - t0:.3f}s"
            )