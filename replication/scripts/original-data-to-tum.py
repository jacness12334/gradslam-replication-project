from scipy.spatial.transform import Rotation
import pathlib
import json
from PIL import Image
import numpy as np

def original_to_tum(indir: pathlib.Path, outdir: pathlib.Path):
    keydir = indir / "keyframes"
    cam_dir   = keydir / "cameras"
    img_dir   = keydir / "images"
    depth_dir = keydir / "depth"

    out_rgb   = outdir / "rgb"
    out_depth = outdir / "depth"
    out_rgb.mkdir(parents=True, exist_ok=True)
    out_depth.mkdir(parents=True, exist_ok=True)
    
    file_assoaciations: list[str] = []
    ground_truths: list[str] = []
    intrinsics_written = False

    for f in cam_dir.iterdir():
        fname: str = f.stem

        file_assoaciations.append(f"{fname} rgb/{fname}.png {fname} depth/{fname}.png")

        current_camera = json.loads(f.read_text())
        current_pose = np.array([
            [current_camera["t_00"], current_camera["t_01"], current_camera["t_02"]], 
            [current_camera["t_10"], current_camera["t_11"], current_camera["t_12"]], 
            [current_camera["t_20"], current_camera["t_21"], current_camera["t_22"]]
        ])
        tx = current_camera["t_03"]
        ty = current_camera["t_13"]
        tz = current_camera["t_23"]
        qx, qy, qz, qw = Rotation.from_matrix(current_pose).as_quat()
        ground_truths.append(f"{fname} {tx:.8f} {ty:.8f} {tz:.8f} "
                    f"{qx:.8f} {qy:.8f} {qz:.8f} {qw:.8f}")

        current_image = Image.open(img_dir / (fname + ".jpg"))
        current_x, current_y = current_image.size
        current_image = current_image.convert("RGB")
        current_image = current_image.resize((640, 480), resample=Image.Resampling.LANCZOS)
        current_image = current_image.save(out_depth / (fname + ".png"))


        current_depth = Image.open(depth_dir / (fname + ".jpg"))
        current_depth = Image.fromarray(np.array(current_depth, dtype=np.uint16), mode="I;16")
        current_depth = current_depth.resize((640, 480), Image.Resampling.NEAREST)
        current_depth.save(out_depth / (fname + ".png"))


        if not intrinsics_written:

            fx = current_camera["fx"] * (640/current_x)
            fy = current_camera["fy"] * (480/current_y)
            cx = current_camera["cx"] * (640/current_x)
            cy = current_camera["cy"] * (480/current_y)
            intrinsics_written = True
            intr_text = (
                f"# fx fy cx cy  (pixels, for 640x480 images)\n"
                f"{fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n"
                f"\n"
                f"# 4x4 intrinsics matrix K\n"
                f"{fx:.6f} 0.000000 {cx:.6f} 0.000000\n"
                f"0.000000 {fy:.6f} {cy:.6f} 0.000000\n"
                f"0.000000 0.000000 1.000000 0.000000\n"
                f"0.000000 0.000000 0.000000 1.000000\n"
            )
            (outdir / "intrinsics.txt").write_text(intr_text)

    (outdir / "associations.txt").write_text("\n".join(file_assoaciations) + "\n")
    (outdir / "groundtruth.txt").write_text(
        "# timestamp tx ty tz qx qy qz qw  (camera-to-world)\n"
        + "\n".join(ground_truths) + "\n"
    )

    print(f"\nDone! Output written to {outdir}")
    print(f"  rgb/        {len(list(out_rgb.glob('*.png')))} PNGs ")
    print(f"  depth/      {len(list(out_depth.glob('*.png')))} PNGs ")
    print(f"  associations.txt")
    print(f"  intrinsics.txt")
    print(f"  groundtruth.txt")

