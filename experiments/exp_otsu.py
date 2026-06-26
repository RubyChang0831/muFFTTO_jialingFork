import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from muFFTTO.otsu import (
    load_npy_image,
    run_otsu_contour_pipeline,
    pack_useful_fields_to_mugrid,
    print_field_summary,
    save_result_csvs,
)


INPUT_RELATIVE = Path("Grain Boundaries Data") / "Green_Jacobi_eta_0.01_w_10.0_p_0.0_final.npy"
OUTPUT_DIRNAME = "output_Otsu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment script for Otsu + contour + muGrid real_field visualization."
    )
    parser.add_argument(
        "--ghosts",
        type=int,
        default=1,
        help="Number of muGrid ghost cells per side (default: 1).",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=20,
        help="Minimum connected-component area kept for region summary (default: 20).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open matplotlib windows; only save figures.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed muGrid storage information.",
    )
    return parser.parse_args()


def ensure_output_dir(simulation_root: Path) -> Path:
    output_dir = simulation_root / OUTPUT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def compute_connected_regions(mask_binary: np.ndarray, min_area: int = 20) -> dict:
    """
    Compute connected components from a binary mask.

    Returns
    -------
    dict with keys:
        num_labels
        labels
        stats
        centroids
        kept_labels
        kept_mask
    """
    binary_u8 = (mask_binary > 0).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_u8,
        connectivity=8,
        ltype=cv2.CV_32S,
    )

    kept_labels = []
    kept_mask = np.zeros_like(binary_u8, dtype=np.uint8)

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area >= min_area:
            kept_labels.append(label_id)
            kept_mask[labels == label_id] = 1

    return {
        "num_labels": int(num_labels),
        "labels": labels,
        "stats": stats,
        "centroids": centroids,
        "kept_labels": kept_labels,
        "kept_mask": kept_mask,
    }


def make_region_summary(region_info: dict) -> list[dict]:
    stats = region_info["stats"]
    centroids = region_info["centroids"]
    kept_labels = region_info["kept_labels"]

    rows = []
    for label_id in kept_labels:
        rows.append(
            {
                "label": int(label_id),
                "area": int(stats[label_id, cv2.CC_STAT_AREA]),
                "left": int(stats[label_id, cv2.CC_STAT_LEFT]),
                "top": int(stats[label_id, cv2.CC_STAT_TOP]),
                "width": int(stats[label_id, cv2.CC_STAT_WIDTH]),
                "height": int(stats[label_id, cv2.CC_STAT_HEIGHT]),
                "cx": float(centroids[label_id, 0]),
                "cy": float(centroids[label_id, 1]),
            }
        )

    rows.sort(key=lambda row: row["area"], reverse=True)
    return rows


def save_region_summary_csv(rows: list[dict], output_dir: Path) -> Path:
    path = output_dir / "grain_boundary_regions.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def print_pipeline_summary(result: dict, region_rows: list[dict]) -> None:
    params = result["params"]

    print("\nPipeline summary")
    print("-" * 72)
    print("shape:", params["shape"])
    print("value_min:", params["value_min"])
    print("value_max:", params["value_max"])
    print("otsu_thresh:", params["otsu_thresh"])
    print("mask_pixel_count:", params["mask_pixel_count"])
    print("contour_pixel_count:", params["contour_pixel_count"])
    print("contour_count:", params["contour_count"])
    print("kept_region_count:", len(region_rows))

    if region_rows:
        print("\nLargest regions")
        print("-" * 72)
        for row in region_rows[:10]:
            print(
                f"label={row['label']:>3d} | "
                f"area={row['area']:>7d} | "
                f"bbox=({row['left']}, {row['top']}, {row['width']}, {row['height']}) | "
                f"centroid=({row['cx']:.1f}, {row['cy']:.1f})"
            )


def save_visualizations(
    data: np.ndarray,
    result: dict,
    region_info: dict,
    output_dir: Path,
) -> dict[str, Path]:
    img_u8 = result["image_u8"]
    blur = result["image_blur"]
    mask_raw = result["mask_raw"]
    mask_binary = result["mask_binary"]
    contour_mask = result["contour"]

    labels = region_info["labels"]
    kept_mask = region_info["kept_mask"]

    fig, axes = plt.subplots(1, 6, figsize=(28, 5))

    axes[0].imshow(
        data,
        cmap="gray",
        origin="lower",
        vmin=float(data.min()),
        vmax=float(data.max()),
    )
    axes[0].set_title("Original Data")

    axes[1].imshow(img_u8, cmap="gray", origin="lower")
    axes[1].set_title("Normalized 8-bit")

    axes[2].imshow(blur, cmap="gray", origin="lower")
    axes[2].set_title("Gaussian Blur")

    axes[3].imshow(mask_raw, cmap="gray", origin="lower")
    axes[3].set_title("Otsu Mask (raw)")

    axes[4].imshow(mask_binary, cmap="gray", origin="lower")
    axes[4].set_title("Mask Binary")

    axes[5].imshow(contour_mask, cmap="gray", origin="lower")
    axes[5].set_title("Contour Mask")

    for ax in axes:
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")

    plt.tight_layout()
    pipeline_png = output_dir / "grain_boundary_mask_contour_result.png"
    plt.savefig(pipeline_png, dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(6, 6))
    plt.imshow(
        data,
        cmap="gray",
        origin="lower",
        vmin=float(data.min()),
        vmax=float(data.max()),
    )
    plt.contour(contour_mask, levels=[0.5], colors="lime", linewidths=0.7)
    plt.title("Original + Contour")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()

    overlay_png = output_dir / "grain_boundary_contour_overlay.png"
    plt.savefig(overlay_png, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].imshow(mask_binary, cmap="gray", origin="lower")
    axes[0].set_title("Binary Regions")

    im = axes[1].imshow(labels, cmap="nipy_spectral", origin="lower")
    axes[1].set_title("Connected Components")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(
        data,
        cmap="gray",
        origin="lower",
        vmin=float(data.min()),
        vmax=float(data.max()),
    )
    axes[2].imshow(
        np.ma.masked_where(kept_mask == 0, kept_mask),
        cmap="autumn",
        alpha=0.55,
        origin="lower",
    )
    axes[2].contour(contour_mask, levels=[0.5], colors="cyan", linewidths=0.5)
    axes[2].set_title("Original + Regions + Contour")

    for ax in axes:
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")

    plt.tight_layout()
    regions_png = output_dir / "grain_boundary_regions.png"
    plt.savefig(regions_png, dpi=150)
    plt.close(fig)

    return {
        "pipeline_png": pipeline_png,
        "overlay_png": overlay_png,
        "regions_png": regions_png,
    }


def maybe_show_figures(
    data: np.ndarray,
    result: dict,
    region_info: dict,
    no_show: bool,
) -> None:
    if no_show:
        return

    labels = region_info["labels"]
    kept_mask = region_info["kept_mask"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].imshow(result["mask_binary"], cmap="gray", origin="lower")
    axes[0].set_title("Mask Binary")

    axes[1].imshow(labels, cmap="nipy_spectral", origin="lower")
    axes[1].set_title("Connected Components")

    axes[2].imshow(
        data,
        cmap="gray",
        origin="lower",
        vmin=float(data.min()),
        vmax=float(data.max()),
    )
    axes[2].imshow(
        np.ma.masked_where(kept_mask == 0, kept_mask),
        cmap="autumn",
        alpha=0.55,
        origin="lower",
    )
    axes[2].contour(result["contour"], levels=[0.5], colors="cyan", linewidths=0.5)
    axes[2].set_title("Overlay")

    plt.tight_layout()
    plt.show()


def main() -> None:
    args = parse_args()

    simulation_root = Path(__file__).resolve().parents[2]
    input_file = simulation_root / INPUT_RELATIVE
    output_dir = ensure_output_dir(simulation_root)

    print("[1] exp_otsu started", flush=True)
    print("[2] simulation_root =", simulation_root, flush=True)
    print("[3] input_file =", input_file, flush=True)
    print("[4] output_dir =", output_dir, flush=True)
    print("[5] exists =", input_file.exists(), flush=True)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    data = load_npy_image(input_file)
    print("[6] data loaded, shape =", data.shape, "dtype =", data.dtype, flush=True)

    start_time = time.perf_counter()
    result = run_otsu_contour_pipeline(data)
    elapsed_time = time.perf_counter() - start_time

    print("[7] pipeline done in", f"{elapsed_time:.4f}", "s", flush=True)
    print("[8] result keys =", list(result.keys()), flush=True)

    packed = pack_useful_fields_to_mugrid(
        result,
        ghosts=args.ghosts,
        verbose=args.verbose,
    )
    print_field_summary(packed)

    csv_paths = save_result_csvs(result, output_dir)

    region_info = compute_connected_regions(
        result["mask_binary"],
        min_area=args.min_area,
    )
    region_rows = make_region_summary(region_info)
    region_csv = save_region_summary_csv(region_rows, output_dir)

    fig_paths = save_visualizations(
        data=data,
        result=result,
        region_info=region_info,
        output_dir=output_dir,
    )

    print_pipeline_summary(result, region_rows)

    print("\nSaved files")
    print("-" * 72)
    print("mask CSV:", csv_paths["mask_binary_csv"])
    print("contour CSV:", csv_paths["contour_csv"])
    print("region CSV:", region_csv)
    print("pipeline figure:", fig_paths["pipeline_png"])
    print("overlay figure:", fig_paths["overlay_png"])
    print("regions figure:", fig_paths["regions_png"])

    maybe_show_figures(
        data=data,
        result=result,
        region_info=region_info,
        no_show=args.no_show,
    )


if __name__ == "__main__":
    main()