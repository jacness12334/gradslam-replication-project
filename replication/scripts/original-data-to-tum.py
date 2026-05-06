from scipy.spatial.transform import Rotation
import pathlib
import json
from Pillow import Image

def original_to_tum(indir: pathlib.Path, outdir: pathlib.Path):
    keydir = indir / "keyframes"
    cam_dir   = keydir / "cameras"
    img_dir   = keydir / "images"
    depth_dir = keydir / "depth"

    out_rgb   = outdir / "rgb"
    out_depth = outdir / "depth"
    out_rgb.mkdir(parents=True, exist_ok=True)
    out_depth.mkdir(parents=True, exist_ok=True)
    
    for f in cam_dir.iterdir():
        fname = f.stem

        current_camera = json.loads(f.read_text())
        current_image = Image.open(img_dir / (fname + ".jpg")).convert("RGB")
        current_depth = Image.open(depth_dir / (fname + ".jpg"))

