import os
import json
from glob import glob

base_dir = os.path.expanduser("~/.objaverse/hf-objaverse-v1/glbs")
glb_paths = glob(os.path.join(base_dir, "**", "*.glb"), recursive=True)
glb_paths = [os.path.abspath(p) for p in glb_paths]
glb_paths.sort()

print(f"找到 {len(glb_paths)} 個 .glb 檔案")
with open("model_paths.json", "w") as f:
    json.dump(glb_paths, f, indent=2)