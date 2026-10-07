import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import glob
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from embed.search import Searcher
from miners.miners import hard_braking

app = FastAPI(
    title="DriveMine",
    description="Scenario mining API for autonomous driving logs",
    version="0.1.0",
)

searcher = Searcher()

samples = pd.concat(
    [
        pd.read_parquet(f)
        for f in sorted(glob.glob("data/parquet/samples/*.parquet"))
    ],
    ignore_index=True,
)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "samples": len(samples),
        "clip_index": searcher.index.ntotal,
    }


@app.get("/search/text")
def search_text(
    q: str = Query(..., min_length=1),
    k: int = Query(5, ge=1, le=50),
):
    result = searcher.text(q, k)

    return {
        "query": q,
        "results": result[
            ["sample_token", "scene", "ts", "cam_path", "score"]
        ].to_dict(orient="records"),
    }


@app.get("/search/similar/{sample_token}")
def search_similar(
    sample_token: str,
    k: int = Query(5, ge=1, le=50),
):
    try:
        result = searcher.similar(sample_token, k)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail="sample_token not found",
        )

    return {
        "sample_token": sample_token,
        "results": result[
            ["sample_token", "scene", "ts", "cam_path", "score"]
        ].to_dict(orient="records"),
    }


@app.get("/mine/hard-braking/{scene}")
def mine_hard_braking(
    scene: str,
    threshold: float = Query(3.0, gt=0),
):
    df = samples[samples.scene == scene].sort_values("ts")

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail="scene not found",
        )

    positions = df[["ex", "ey"]].to_numpy(dtype=np.float64)
    timestamps = df["ts"].to_numpy(dtype=np.float64)

    events = hard_braking(
        np.column_stack([positions, np.zeros(len(positions))]),
        timestamps,
        threshold,
    )

    return {
        "scene": scene,
        "threshold": threshold,
        "events": [
            {
                "sample_token": df.iloc[int(i)].sample_token,
                "timestamp": int(df.iloc[int(i)].ts),
                "index": int(i),
            }
            for i in events
        ],
    }
