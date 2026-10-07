# DriveMine

## Scenario Mining Platform for Autonomous Driving Logs

DriveMine is a scenario mining platform built over the nuScenes mini dataset. It combines parallel data ingestion, C++/pybind11 scenario detection kernels, CLIP semantic search, FAISS vector retrieval, REST APIs, and replayable JSON scenario export.

## Architecture

```text
nuScenes
    |
    v
Parallel ingestion
    |
    v
Parquet
    |
    +----> C++ / NumPy scenario miners
    |
    +----> CLIP image embeddings
                 |
                 v
              FAISS
                 |
                 v
              FastAPI
                 |
        +--------+--------+
        |                 |
   Text search      Similar frames
        |
        v
   Scenario export
        |
        v
   Replayable JSON

