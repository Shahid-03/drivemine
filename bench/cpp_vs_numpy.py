import time
import numpy as np
import cpp_miners


N = 5_000_000

rng = np.random.default_rng(42)

dt = 0.5
timestamps = np.arange(N, dtype=np.float64) * dt * 1e6

velocity = 15.0 + rng.normal(0, 0.5, N)

x = np.cumsum(velocity * dt)
y = np.cumsum(rng.normal(0, 0.05, N))

x = np.ascontiguousarray(x)
y = np.ascontiguousarray(y)
timestamps = np.ascontiguousarray(timestamps)


def hard_braking_np(x, y, ts, threshold=3.0):
    dt = np.diff(ts) / 1e6
    v = np.hypot(np.diff(x), np.diff(y)) / dt
    a = np.diff(v) / dt[1:]
    return np.where(a < -threshold)[0] + 2


def benchmark(fn, name, repeats=5):
    times = []

    fn()

    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)

    best = min(times)

    print(
        f"{name}: "
        f"best={best:.6f}s "
        f"mean={np.mean(times):.6f}s "
        f"events={len(result)}"
    )

    return best, result


numpy_time, numpy_result = benchmark(
    lambda: hard_braking_np(x, y, timestamps),
    "numpy",
)

cpp_time, cpp_result = benchmark(
    lambda: cpp_miners.hard_braking(x, y, timestamps),
    "cpp",
)

print(f"numpy_events={len(numpy_result)}")
print(f"cpp_events={len(cpp_result)}")
print(f"speedup={numpy_time / cpp_time:.2f}x")
print(f"rows={N}")
