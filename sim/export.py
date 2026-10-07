import json
import os
import pandas as pd


def load_scene(scene):
    sample_path = f"data/parquet/samples/{scene}.parquet"
    ann_path = f"data/parquet/annotations/{scene}.parquet"

    if not os.path.exists(sample_path):
        raise ValueError(f"scene not found: {scene}")

    samples = pd.read_parquet(sample_path).sort_values("ts").reset_index(drop=True)
    annotations = pd.read_parquet(ann_path)

    return samples, annotations


def export_scenario(scene, event_index, output_dir="data/scenarios"):
    samples, annotations = load_scene(scene)

    if event_index < 0 or event_index >= len(samples):
        raise ValueError("event_index out of range")

    event = samples.iloc[event_index]

    event_annotations = annotations[
        annotations.sample_token == event.sample_token
    ]

    objects = []

    for _, row in event_annotations.iterrows():
        objects.append({
            "instance": row.instance,
            "category": row.category,
            "position": {
                "x": float(row.x),
                "y": float(row.y),
                "z": float(row.z),
            },
            "size": {
                "width": float(row.w),
                "length": float(row.l),
                "height": float(row.h),
            },
            "velocity": {
                "x": float(row.vx),
                "y": float(row.vy),
            },
            "lidar_points": int(row.lidar_pts),
        })

    scenario = {
        "schema_version": "1.0",
        "scene": scene,
        "event": {
            "sample_token": event.sample_token,
            "timestamp": int(event.ts),
            "index": int(event_index),
        },
        "ego_pose": {
            "position": {
                "x": float(event.ex),
                "y": float(event.ey),
                "z": float(event.ez),
            },
            "rotation": {
                "qw": float(event.qw),
                "qx": float(event.qx),
                "qy": float(event.qy),
                "qz": float(event.qz),
            },
        },
        "camera": {
            "channel": "CAM_FRONT",
            "path": event.cam_path,
        },
        "objects": objects,
    }

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        f"{scene}_{event_index}.json",
    )

    with open(output_path, "w") as f:
        json.dump(scenario, f, indent=2)

    return output_path, scenario
