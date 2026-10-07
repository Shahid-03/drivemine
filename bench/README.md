# Benchmarks

## Parallel ingestion

Dataset: nuScenes mini, 404 CAM_FRONT samples.

| Workers | Time | Samples/s |
|---:|---:|---:|
| 1 | 3.6s | 113.2 |
| 2 | 2.1s | 193.3 |
| 4 | 2.8s | 144.1 |
| 8 | 4.9s | 82.3 |

Best configuration: 2 workers, 193.3 samples/s.

## C++ vs NumPy

Workload: 5,000,000 synthetic trajectory rows.



| Implementation | Best time |
|---|---:|
| NumPy | 21.901 ms |
| C++/pybind11 | 15.171 ms |

Speedup: **1.44x**

Speedup: **1.44x**

Both implementations detected 84,750 events.

## API latency

Endpoint: `GET /search/text?q=night%20driving&k=5`

20 requests against local FastAPI/Uvicorn:

| Metric | Latency |
|---|---:|
| Minimum | 14.7 ms |
| Mean | 55.9 ms |
| p95 | 67.5 ms |
