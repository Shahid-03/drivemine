import sys, time, os
from multiprocessing import Pool
import pandas as pd
from nuscenes.nuscenes import NuScenes

ROOT, VER, OUT = "data/nuscenes", "v1.0-mini", "data/parquet"
nusc = None

def init():
    global nusc
    nusc = NuScenes(version=VER, dataroot=ROOT, verbose=False)

def process_scene(scene_token):
    scene = nusc.get("scene", scene_token)
    samples, anns = [], []
    tok = scene["first_sample_token"]
    while tok:
        s = nusc.get("sample", tok)
        sd = nusc.get("sample_data", s["data"]["CAM_FRONT"])
        p = nusc.get("ego_pose", sd["ego_pose_token"])
        samples.append(dict(
            sample_token=tok, scene=scene["name"], ts=s["timestamp"],
            ex=p["translation"][0], ey=p["translation"][1], ez=p["translation"][2],
            qw=p["rotation"][0], qx=p["rotation"][1], qy=p["rotation"][2], qz=p["rotation"][3],
            cam_path=sd["filename"]))
        for a in s["anns"]:
            r = nusc.get("sample_annotation", a)
            v = nusc.box_velocity(a)
            anns.append(dict(
                sample_token=tok, scene=scene["name"], instance=r["instance_token"],
                category=r["category_name"], x=r["translation"][0], y=r["translation"][1],
                z=r["translation"][2], w=r["size"][0], l=r["size"][1], h=r["size"][2],
                vx=v[0], vy=v[1], lidar_pts=r["num_lidar_pts"]))
        tok = s["next"]
    for name, rows in (("samples", samples), ("annotations", anns)):
        os.makedirs(f"{OUT}/{name}", exist_ok=True)
        pd.DataFrame(rows).to_parquet(f"{OUT}/{name}/{scene['name']}.parquet")
    return len(samples)

if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    tmp = NuScenes(version=VER, dataroot=ROOT, verbose=False)
    tokens = [s["token"] for s in tmp.scene]
    t0 = time.time()
    with Pool(workers, initializer=init) as pool:
        n = sum(pool.map(process_scene, tokens))
    dt = time.time() - t0
    print(f"workers={workers} samples={n} time={dt:.1f}s samples/s={n/dt:.1f}")