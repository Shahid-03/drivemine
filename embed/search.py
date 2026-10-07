import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import torch
import faiss
import open_clip

dev = "mps" if torch.backends.mps.is_available() else "cpu"


class Searcher:
    def __init__(self):
        self.model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="laion2b_s34b_b79k",
        )
        self.model = self.model.to(dev).eval()

        self.tok = open_clip.get_tokenizer("ViT-B-32")
        self.index = faiss.read_index("data/clip.faiss")
        self.meta = pd.read_parquet("data/index_meta.parquet")

    def _fmt(self, scores, ids):
        out = self.meta.iloc[ids].copy()
        out["score"] = scores
        return out

    def text(self, q, k=5):
        with torch.no_grad():
            e = self.model.encode_text(
                self.tok([q]).to(dev)
            )
            e = (
                e / e.norm(dim=-1, keepdim=True)
            ).cpu().numpy().astype("float32")

        scores, ids = self.index.search(e, k)
        return self._fmt(scores[0], ids[0])

    def similar(self, sample_token, k=5):
        matches = self.meta.index[
            self.meta.sample_token == sample_token
        ]

        if len(matches) == 0:
            raise ValueError(f"Unknown sample_token: {sample_token}")

        i = int(matches[0])

        vector = self.index.reconstruct(i).reshape(1, -1)
        scores, ids = self.index.search(vector, k + 1)

        return self._fmt(scores[0][1:], ids[0][1:])


if __name__ == "__main__":
    searcher = Searcher()

    for query in [
        "a pedestrian crossing the road",
        "night driving",
        "rain",
        "a truck ahead",
    ]:
        print(f"\n=== {query} ===")
        print(
            searcher.text(query)[
                ["scene", "cam_path", "score"]
            ].to_string(index=False)
        )
