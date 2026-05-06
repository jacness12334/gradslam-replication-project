from scipy.spatial.transform import Rotation
import pathlib
import json
from PIL import Image
import numpy as np
import torch
from gradslam.datasets import TUM
from gradslam.structures.rgbdimages import RGBDImages
from gradslam.slam.icpslam import ICPSLAM
from gradslam.slam.pointfusion import PointFusion
from typing import Tuple, Any
from gradslam.structures.pointclouds import Pointclouds
import time


def original_to_tum(indir, outdir):
    keydir = indir / "keyframes"
    cam_dir   = keydir / "cameras"
    img_dir   = keydir / "images"
    depth_dir = keydir / "depth"

    tum_seq_dir = outdir / "rgbd_dataset_freiburg1_replica"
    out_rgb   = tum_seq_dir / "rgb"
    out_depth = tum_seq_dir / "depth"
    out_rgb.mkdir(parents=True, exist_ok=True)
    out_depth.mkdir(parents=True, exist_ok=True)
    
    file_assoaciations = []
    rgb_files_txt_lines = []
    depth_files_txt_lines = []
    ground_truths = []
    intrinsics_written = False

    for f in sorted(cam_dir.iterdir()):
        fname = f.stem
        
        rgb_files_txt_lines.append(f"{fname} rgb/{fname}.png")
        depth_files_txt_lines.append(f"{fname} depth/{fname}.png")
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
        current_image.save(out_rgb / (fname + ".png"))


        current_depth = Image.open(depth_dir / (fname + ".png"))
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
            (tum_seq_dir / "intrinsics.txt").write_text(intr_text)

    (tum_seq_dir / "rgb.txt").write_text(
        "# timestamp filename\n" + "\n".join(rgb_files_txt_lines) + "\n"
    )
    (tum_seq_dir / "depth.txt").write_text(
        "# timestamp filename\n" + "\n".join(depth_files_txt_lines) + "\n"
    )
    (tum_seq_dir / "associations.txt").write_text(
        "# rgb_timestamp rgb_filename depth_timestamp depth_filename\n" + "\n".join(file_assoaciations) + "\n"
    )
    (tum_seq_dir / "groundtruth.txt").write_text(
        "# timestamp tx ty tz qx qy qz qw  (camera-to-world)\n"
        + "\n".join(ground_truths) + "\n"
    )

    print(f"\nDone! Output written to {tum_seq_dir}")
    print(f"  rgb/        {len(list(out_rgb.glob('*.png')))} PNGs ")
    print(f"  depth/      {len(list(out_depth.glob('*.png')))} PNGs ")
    print(f"  associations.txt")
    print(f"  intrinsics.txt")
    print(f"  groundtruth.txt")

    return tum_seq_dir.parent, len(rgb_files_txt_lines)


def _read_intrinsics(seq_dir, device):
    """
    Read fx fy cx cy from our custom intrinsics.txt and return a (1,1,4,4)
    camera-intrinsics tensor in the format gradslam expects.
    """
    # intrinsics.txt is in the rgbd_dataset_freiburg1_replica subdirectory
    intrinsics_path = seq_dir / "rgbd_dataset_freiburg1_replica" / "intrinsics.txt"
    lines = [l for l in intrinsics_path.read_text().splitlines()
             if l and not l.startswith("#")]
    fx, fy, cx, cy = map(float, lines[0].split())
    K = torch.tensor([[fx, 0., cx, 0.],
                      [0., fy, cy, 0.],
                      [0.,  0., 1., 0.],
                      [0.,  0., 0., 1.]], dtype=torch.float32, device=device)
    return K.unsqueeze(0).unsqueeze(0)   # (1, 1, 4, 4)

def print_results(name, pointclouds, recovered_poses, elapsed):
    n_pts = pointclouds.num_points_per_pointcloud[0].item()
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  Map points     : {n_pts:,}")
    print(f"  Poses shape    : {tuple(recovered_poses.shape)}")
    print(f"  Elapsed        : {elapsed:.1f}s")
    # Print first and last recovered pose translation (x,y,z)
    t0 = recovered_poses[0, 0, :3, 3].tolist()
    tn = recovered_poses[0, -1,:3, 3].tolist()
    print(f"  First pose t   : [{t0[0]:.3f}, {t0[1]:.3f}, {t0[2]:.3f}]")
    print(f"  Last  pose t   : [{tn[0]:.3f}, {tn[1]:.3f}, {tn[2]:.3f}]")
    print(f"{'─'*50}")

def run_ICP(device, seqlen, rgbdimages_no_poses, slam_grad_icp):
    
    t0 = time.time()
    pointclouds_odom = Pointclouds(device=device)
    prev_frame = None

    for i in range(seqlen):
        live_frame = rgbdimages_no_poses[:, i : i + 1]   # slice → shape (1,1,480,640)
        pointclouds_odom, _ = slam_grad_icp.step(
            pointclouds_odom,
            live_frame,
            prev_frame,        # None on first frame → uses the frame's own pose
        )
        prev_frame = live_frame

    # Collect all poses back from the rgbdimages object
    recovered_poses_odom = rgbdimages_no_poses.poses

    print_results("ICP-Odometry (frame-to-frame)", pointclouds_odom,
                recovered_poses_odom, time.time() - t0)

def run_grad_ICP(device, seqlen, rgbdimages_no_poses, slam_grad_icp):
    
    t0 = time.time()
    pointclouds_odom = Pointclouds(device=device)
    prev_frame = None

    for i in range(seqlen):
        live_frame = rgbdimages_no_poses[:, i : i + 1]   # slice → shape (1,1,480,640)
        pointclouds_odom, _ = slam_grad_icp.step(
            pointclouds_odom,
            live_frame,
            prev_frame,        # None on first frame → uses the frame's own pose
        )
        prev_frame = live_frame

    # Collect all poses back from the rgbdimages object
    recovered_poses_odom = rgbdimages_no_poses.poses

    print_results("∇ICP-Odometry (frame-to-frame)", pointclouds_odom,
                recovered_poses_odom, time.time() - t0)

    
def run_ICP_SLAM(device, rgbdimages_with_poses):

    slam_icp_slam = ICPSLAM(
        odom     = "gt",
        dsratio  = 4,
        numiters = 20,
        damp     = 1e-8,
        device   = device,
    )

    t0 = time.time()
    # The one-liner forward() call handles the frame-to-model loop internally.
    # It aligns each frame against the running global map (not the previous frame).
    pointclouds_slam, recovered_poses_slam = slam_icp_slam(rgbdimages_with_poses)

    print_results("∇ICP-SLAM (frame-to-model)", pointclouds_slam,
                recovered_poses_slam, time.time() - t0)
    
    pointclouds_slam.plotly(0).show()

def run_point_fusion(device, rgbdimages_no_poses):
    slam_pf = PointFusion(
        dist_th  = 0.05,    # merge surfels within 5cm of each other
        angle_th = 20,      # merge surfels within 20° normal angle
        sigma    = 0.6,     # Gaussian width for soft association (paper default)
        dsratio  = 4,
        numiters = 20,
        damp     = 1e-8,
        device   = device,
    )

    t0 = time.time()
    pointclouds_pf, recovered_poses_pf = slam_pf(rgbdimages_no_poses)

    print_results("∇PointFusion (surfel)", pointclouds_pf,
                recovered_poses_pf, time.time() - t0)
    
    pointclouds_pf.plotly(0).show()

# tum dir is output dir, img count is returned from original to tum, device is 'cpu' or 'cuda'
def run_all_techniques(tum_dir, img_count):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'═'*50}")
    print(f"  Running all techniques on: {tum_dir.name}")
    print(f"  Device: {device}  |  Frames: {img_count}")
    print(f"{'═'*50}")


    dataset = TUM(
        basedir    = str(tum_dir.absolute()),
        seqlen     = img_count,
        height     = 480,
        width      = 640,
        normalize_color = True,      # converts uint8 → float [0,1]
        return_depth      = True,
        return_intrinsics = True,
        return_pose       = True,
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    colors, depths, intrinsics, poses, *_ = next(iter(loader))
    colors     = colors.to(device)
    depths     = depths.to(device)
    intrinsics = intrinsics.to(device)
    poses      = poses.to(device)

    # Use custom intrinsics read from the dataset
    intrinsics = _read_intrinsics(tum_dir, device)

    rgbdimages_with_poses = RGBDImages(
        rgb_image   = colors,
        depth_image = depths,
        intrinsics  = intrinsics,
        poses       = poses, 
    ).to(device)

    rgbdimages_no_poses = RGBDImages(
        rgb_image   = colors,
        depth_image = depths,
        intrinsics  = intrinsics,
    ).to(device)

    slam_icp = ICPSLAM(
        odom     = 'icp',         # 'gradicp' → ∇LM;  'icp' → classic LM
        dsratio  = 4,            # downsample frames by 4x before ICP (speeds up)
        numiters = 20,           # LM iterations per frame
        damp     = 1e-8,         # LM damping floor
        device   = device,
    )

    run_ICP(device=device, seqlen=img_count, rgbdimages_no_poses=rgbdimages_with_poses, slam_grad_icp=slam_icp)

    slam_grad_icp = ICPSLAM(
        odom     = 'gradicp',         # 'gradicp' → ∇LM;  'icp' → classic LM
        dsratio  = 4,            # downsample frames by 4x before ICP (speeds up)
        numiters = 20,           # LM iterations per frame
        damp     = 1e-8,         # LM damping floor
        device   = device,
    )

    run_grad_ICP(device=device, seqlen=img_count, rgbdimages_no_poses=rgbdimages_with_poses, slam_grad_icp=slam_grad_icp)
    
    rgbdimages_no_poses = RGBDImages(
        rgb_image   = colors,
        depth_image = depths,
        intrinsics  = intrinsics,
    ).to(device)

    run_ICP_SLAM(device=device, rgbdimages_with_poses=rgbdimages_with_poses)

    run_point_fusion(device=device, rgbdimages_no_poses=rgbdimages_no_poses)


def main():
    script_dir = pathlib.Path(__file__).resolve().parent
    data_dir   = script_dir.parent / "data"         # .../replication/data
    if not data_dir.exists():
        data_dir = script_dir / "data"               # fallback to .../replication/scripts/data
    original_dir = data_dir / "original"
    if not original_dir.exists():
        print(f"Data directory not found: {original_dir}")
        return
    # make new tum dir for converted data
    tum_root     = data_dir / "tum"
    tum_root.mkdir(parents=True, exist_ok=True)
 
    scenes = sorted(p for p in original_dir.iterdir() if p.is_dir())
    if not scenes:
        print(f"No scene folders found under {original_dir}")
        return
 
    print(f"Found {len(scenes)} scene(s): {[s.name for s in scenes]}")
 
    for scene_dir in scenes:
        print(f"\n{'━'*60}")
        print(f"  Scene: {scene_dir.name}")
        print(f"{'━'*60}")
 
        # Each scene becomes its own TUM sequence subdirectory.
        seq_out = tum_root / scene_dir.name
 
        try:
            seq_dir, img_count = original_to_tum(scene_dir, seq_out)
        except Exception as exc:
            print(f"  [SKIP] Conversion failed: {exc}")
            continue
 
        if img_count == 0:
            print("  [SKIP] No frames found after conversion.")
            continue
 
        try:
            run_all_techniques(seq_dir, img_count)
        except Exception as exc:
            print(f"  [SKIP] SLAM failed: {exc}")
            continue
 
    print("\nAll scenes processed.")
 
 
if __name__ == "__main__":
    main()
