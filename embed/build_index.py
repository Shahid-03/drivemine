import glob
import numpy as np
import pandas as pd
import torch
import faiss
import open_clip
from PIL import Image

ROOT = "data/nuscenes"

dev = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device={dev}")

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k",
)
model = model.to(dev).eval()

files = sorted(glob.glob("data/parquet/samples/*.parquet"))
s = pd.concat(
    [pd.read_parquet(f) for f in files],
    ignore_index=True,
)

print(f"images={len(s)}")

embs = []

with torch.no_grad():
    for i in range(0, len(s), 32):
        paths = s.cam_path.iloc[i:i + 32]

        batch = torch.stack([
            preprocess(
                Image.open(f"{ROOT}/{p}").convert("RGB")
            )
            for p in paths
        ]).to(dev)

        e = model.encode_image(batch)
        e = e / e.norm(dim=-1, keepdim=True)
        embs.append(e.cpu().numpy())

        print(f"processed={min(i + 32, len(s))}/{len(s)}")

embs = np.concatenate(embs).astype("float32")

index = faiss.IndexFlatIP(embs.shape[1])
index.add(embs)

faiss.write_index(index, "data/clip.faiss")

s[["sample_token", "scene", "ts", "cam_path"]].to_parquet(
    "data/index_meta.parquet"
)

print(f"indexed={index.ntotal}")
print(f"embedding_dim={embs.shape[1]}")
print("saved=data/clip.faiss")
print("saved=data/index_meta.parquet")
