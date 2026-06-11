"""Extract all images from a Jupyter notebook into article/images/."""
import argparse
import base64
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser(description="Extract plots from a .ipynb notebook.")
parser.add_argument("notebook", help="Path to the .ipynb file (or just the stem name inside notebooks/)")
parser.add_argument("--out", default="article/images", help="Output directory (default: article/images)")
parser.add_argument("--prefix", default="", help="Filename prefix for saved images")
args = parser.parse_args()

nb_path = Path(args.notebook)
if not nb_path.exists():
    # try notebooks/<name>.ipynb relative to repo root
    nb_path = Path("notebooks") / (args.notebook if args.notebook.endswith(".ipynb") else args.notebook + ".ipynb")
if not nb_path.exists():
    raise FileNotFoundError(f"Notebook not found: {nb_path}")

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)

prefix = args.prefix or nb_path.stem + "_"

img_count = 0
for i, cell in enumerate(nb["cells"]):
    for output in cell.get("outputs", []):
        if output.get("output_type") not in ("display_data", "execute_result"):
            continue
        data = output.get("data", {})
        for fmt, content in data.items():
            if "png" not in fmt and "jpeg" not in fmt:
                continue
            ext = "png" if "png" in fmt else "jpg"
            raw = content if isinstance(content, str) else "".join(content)
            img_bytes = base64.b64decode(raw)
            fname = out_dir / f"{prefix}cell{i:02d}_{img_count:02d}.{ext}"
            fname.write_bytes(img_bytes)
            print(f"  saved {fname}  ({len(img_bytes) // 1024} KB)")
            img_count += 1

print(f"Total: {img_count} images extracted from {nb_path}")
