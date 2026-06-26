from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import muGrid


# ============================================================
# Basic I/O
# ============================================================


def load_npy_image(path: str | os.PathLike[str], dtype=np.float32) -> np.ndarray:
    """
    Function that loads a 2D image stored in .npy format.

    Parameters
    ----------
    path : str or PathLike
        Path to a .npy file containing a 2D array.
    dtype : numpy dtype
        Target dtype after loading.

    Returns
    -------
    data : numpy.ndarray
        Loaded 2D array with shape [nx, ny].

    Raises
    ------
    ValueError
        If the loaded array is not 2D.
    """
    data = np.load(path).astype(dtype)
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {data.shape}.")
    return data


# ============================================================
# Image processing
# ============================================================


def normalize_to_u8(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Function that normalizes a float image to [0, 1] and uint8 [0, 255].

    Parameters
    ----------
    data : numpy.ndarray
        Input 2D image.

    Returns
    -------
    img_norm : numpy.ndarray
        Float image normalized to [0, 1].
    img_u8 : numpy.ndarray
        Unsigned 8-bit image normalized to [0, 255].
    """
    data = np.asarray(data, dtype=np.float32)
    img_norm = (data - data.min()) / (data.max() - data.min() + 1e-12)
    img_u8 = (img_norm * 255).astype(np.uint8)
    return img_norm, img_u8


def gaussian_blur_u8(img_u8: np.ndarray, ksize: int = 7, sigma: float = 1.5) -> np.ndarray:
    """
    Function that applies Gaussian blur to an 8-bit image.

    Parameters
    ----------
    img_u8 : numpy.ndarray
        Input uint8 image.
    ksize : int
        Odd kernel size.
    sigma : float
        Gaussian sigma.

    Returns
    -------
    blur : numpy.ndarray
        Blurred uint8 image.
    """
    if ksize % 2 == 0:
        raise ValueError("ksize must be odd for Gaussian blur.")
    return cv2.GaussianBlur(img_u8, (ksize, ksize), sigma)


def otsu_threshold_dark_foreground(img_u8: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Function that applies Otsu thresholding with inversion.

    This is useful when dark grain boundaries should become foreground.

    Parameters
    ----------
    img_u8 : numpy.ndarray
        Input uint8 image.

    Returns
    -------
    otsu_thresh : float
        Otsu threshold value.
    mask_raw : numpy.ndarray
        Raw binary mask with values {0, 255}.
    """
    otsu_thresh, mask_raw = cv2.threshold(
        img_u8,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    return float(otsu_thresh), mask_raw


def clean_binary_mask(
    mask_raw: np.ndarray,
    kernel_size: int = 3,
    open_iterations: int = 1,
    close_iterations: int = 1,
) -> dict[str, np.ndarray]:
    """
    Function that applies opening and closing to a binary mask.

    Parameters
    ----------
    mask_raw : numpy.ndarray
        Raw binary mask with values {0, 255}.
    kernel_size : int
        Size of the square morphology kernel.
    open_iterations : int
        Number of opening iterations.
    close_iterations : int
        Number of closing iterations.

    Returns
    -------
    cleaned : dict
        Dictionary with keys:
        - "mask_open"
        - "mask_clean"
        - "mask_binary"
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_open = cv2.morphologyEx(mask_raw, cv2.MORPH_OPEN, kernel, iterations=open_iterations)
    mask_clean = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
    mask_binary = (mask_clean > 0).astype(np.uint8)

    return {
        "mask_open": mask_open,
        "mask_clean": mask_clean,
        "mask_binary": mask_binary,
    }


def find_external_contour_mask(mask_binary: np.ndarray) -> dict[str, Any]:
    """
    Function that finds external contours and rasterizes them into a 0/1 contour mask.

    Parameters
    ----------
    mask_binary : numpy.ndarray
        Binary mask with values {0, 1}.

    Returns
    -------
    contour_info : dict
        Dictionary with keys:
        - "contours"
        - "hierarchy"
        - "contour_mask"
        - "contour_pixel_count"
        - "contour_count"
    """
    contours, hierarchy = cv2.findContours(
        (mask_binary * 255).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    contour_mask = np.zeros_like(mask_binary, dtype=np.uint8)
    cv2.drawContours(contour_mask, contours, -1, 1, thickness=1)

    return {
        "contours": contours,
        "hierarchy": hierarchy,
        "contour_mask": contour_mask,
        "contour_pixel_count": int(contour_mask.sum()),
        "contour_count": int(len(contours)),
    }


def run_otsu_contour_pipeline(
    data: np.ndarray,
    blur_ksize: int = 7,
    blur_sigma: float = 1.5,
    morph_kernel_size: int = 3,
    open_iterations: int = 1,
    close_iterations: int = 1,
) -> dict[str, Any]:
    """
    Function that runs the full image-to-mask-to-contour pipeline on a 2D image.

    Parameters
    ----------
    data : numpy.ndarray
        Input 2D image.
    blur_ksize : int
        Gaussian blur kernel size.
    blur_sigma : float
        Gaussian blur sigma.
    morph_kernel_size : int
        Morphology kernel size.
    open_iterations : int
        Opening iterations.
    close_iterations : int
        Closing iterations.

    Returns
    -------
    result : dict
        Dictionary containing processed arrays and scalar metadata.
    """
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D input data, got shape {data.shape}.")

    img_norm, img_u8 = normalize_to_u8(data)
    blur = gaussian_blur_u8(img_u8, ksize=blur_ksize, sigma=blur_sigma)
    otsu_thresh, mask_raw = otsu_threshold_dark_foreground(blur)

    cleaned = clean_binary_mask(
        mask_raw,
        kernel_size=morph_kernel_size,
        open_iterations=open_iterations,
        close_iterations=close_iterations,
    )
    contour_info = find_external_contour_mask(cleaned["mask_binary"])

    return {
        "image_raw": data,
        "image_norm": img_norm,
        "image_u8": img_u8,
        "image_blur": blur,
        "mask_raw": (mask_raw > 0).astype(np.uint8),
        "mask_open": (cleaned["mask_open"] > 0).astype(np.uint8),
        "mask_clean": (cleaned["mask_clean"] > 0).astype(np.uint8),
        "mask_binary": cleaned["mask_binary"],
        "contour": contour_info["contour_mask"],
        "params": {
            "shape": tuple(data.shape),
            "value_min": float(data.min()),
            "value_max": float(data.max()),
            "blur_ksize": int(blur_ksize),
            "blur_sigma": float(blur_sigma),
            "morph_kernel_size": int(morph_kernel_size),
            "open_iterations": int(open_iterations),
            "close_iterations": int(close_iterations),
            "otsu_thresh": float(otsu_thresh),
            "mask_pixel_count": int(cleaned["mask_binary"].sum()),
            "contour_pixel_count": int(contour_info["contour_pixel_count"]),
            "contour_count": int(contour_info["contour_count"]),
        },
    }


# ============================================================
# muGrid helpers
# ============================================================


def make_mugrid_decomposition(
    nx: int,
    ny: int,
    ghosts: int = 1,
):
    """
    Function that creates a simple single-process muGrid Cartesian decomposition.

    Parameters
    ----------
    nx : int
        Number of domain grid points in x-direction.
    ny : int
        Number of domain grid points in y-direction.
    ghosts : int
        Number of ghost points per side.

    Returns
    -------
    comm : muGrid.Communicator
        muGrid communicator.
    decomp : muGrid.CartesianDecomposition
        Cartesian decomposition.
    """
    comm = muGrid.Communicator()
    decomp = muGrid.CartesianDecomposition(
        communicator=comm,
        nb_domain_grid_pts=(nx, ny),
        nb_subdivisions=(1, 1),
        nb_ghosts_left=(ghosts, ghosts),
        nb_ghosts_right=(ghosts, ghosts),
    )
    return comm, decomp


def _as_scalar_field_array(arr: np.ndarray) -> np.ndarray:
    """
    Function that converts a 2D scalar array to shape [1, nx, ny].

    Parameters
    ----------
    arr : numpy.ndarray
        Input scalar array with shape [nx, ny].

    Returns
    -------
    arr3 : numpy.ndarray
        Output array with shape [1, nx, ny].
    """
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D scalar field array, got shape {arr.shape}.")
    return arr[None, ...]


def _copy_array_into_field_p(field, arr: np.ndarray) -> None:
    """
    Function that copies a NumPy array into a muGrid field through field.p.

    Parameters
    ----------
    field : muGrid field
        Target field.
    arr : numpy.ndarray
        Array to be copied.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If arr.shape does not match field.p shape.
    """
    arr = np.asarray(arr)
    target = np.asarray(field.p)
    if target.shape != arr.shape:
        raise ValueError(
            f"Shape mismatch: field.p shape {target.shape}, arr shape {arr.shape}."
        )
    target[...] = arr


def make_numpy_field_bundle(
    result: dict[str, Any],
    store_intermediate: bool = True,
) -> dict[str, np.ndarray]:
    """
    Function that converts pipeline outputs into muGrid-ready NumPy arrays.

    Scalar fields are stored with shape [1, nx, ny], matching the style used
    in the successful circle_muGrid_real_field implementation.

    Parameters
    ----------
    result : dict
        Result from run_otsu_contour_pipeline.
    store_intermediate : bool
        If True, also include intermediate images and masks.

    Returns
    -------
    bundle : dict[str, numpy.ndarray]
        Dictionary of arrays with shape [1, nx, ny].
    """
    useful = {
        "mask_binary": _as_scalar_field_array(result["mask_binary"].astype(np.float64)),
        "contour": _as_scalar_field_array(result["contour"].astype(np.float64)),
    }

    if store_intermediate:
        useful = {
            "image_raw": _as_scalar_field_array(result["image_raw"].astype(np.float64)),
            "image_norm": _as_scalar_field_array(result["image_norm"].astype(np.float64)),
            "image_u8": _as_scalar_field_array(result["image_u8"].astype(np.float64)),
            "image_blur": _as_scalar_field_array(result["image_blur"].astype(np.float64)),
            "mask_raw": _as_scalar_field_array(result["mask_raw"].astype(np.float64)),
            "mask_open": _as_scalar_field_array(result["mask_open"].astype(np.float64)),
            "mask_clean": _as_scalar_field_array(result["mask_clean"].astype(np.float64)),
            **useful,
        }

    return useful


def pack_numpy_fields_to_mugrid(
    numpy_bundle: dict[str, np.ndarray],
    ghosts: int = 1,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Function that packs NumPy arrays into muGrid real_field containers.

    Parameters
    ----------
    numpy_bundle : dict
        Dictionary of arrays with shape [ncomp, nx, ny].
    ghosts : int
        Number of ghost points per side.
    verbose : bool
        If True, print storage information.

    Returns
    -------
    bundle : dict
        Dictionary containing:
        - "numpy"
        - "fields"
        - "report"
        - "decomposition"
        - "communicator"
    """
    if not numpy_bundle:
        raise ValueError("numpy_bundle must not be empty.")

    first = np.asarray(next(iter(numpy_bundle.values())))
    if first.ndim != 3:
        raise ValueError(f"Expected first array to have shape [ncomp, nx, ny], got {first.shape}.")

    _, nx, ny = first.shape
    comm, decomp = make_mugrid_decomposition(nx=nx, ny=ny, ghosts=ghosts)

    fields = {}
    report = {}

    for name, arr in numpy_bundle.items():
        arr = np.asarray(arr)
        if arr.ndim != 3:
            raise ValueError(f"Array '{name}' has shape {arr.shape}, expected [ncomp, nx, ny].")
        if arr.shape[1:] != (nx, ny):
            raise ValueError(f"Array '{name}' has shape {arr.shape}, expected (*, {nx}, {ny}).")

        ncomp = arr.shape[0]
        field = decomp.real_field(name, components=(ncomp,))
        _copy_array_into_field_p(field, arr)

        fields[name] = field
        report[name] = {
            "status": "ok",
            "shape": tuple(arr.shape),
            "dtype": str(arr.dtype),
            "field_p_shape": tuple(np.asarray(field.p).shape),
            "field_pg_shape": tuple(np.asarray(field.pg).shape),
        }

        if verbose:
            print(
                f"[pack] {name}: arr_shape={arr.shape}, "
                f"field_p_shape={np.asarray(field.p).shape}, "
                f"dtype={arr.dtype}",
                flush=True,
            )

    return {
        "numpy": numpy_bundle,
        "fields": fields,
        "report": report,
        "decomposition": decomp,
        "communicator": comm,
    }


def pack_otsu_results_to_mugrid(
    result: dict[str, Any],
    ghosts: int = 1,
    store_intermediate: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Function that converts the Otsu/contour result to NumPy and muGrid fields.
    """
    numpy_bundle = make_numpy_field_bundle(result, store_intermediate=store_intermediate)
    packed = pack_numpy_fields_to_mugrid(numpy_bundle, ghosts=ghosts, verbose=verbose)
    packed["meta"] = result["params"]
    return packed


def pack_useful_fields_to_mugrid(
    result: dict[str, Any],
    ghosts: int = 1,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Function that packs only the most useful outputs for downstream usage.

    Stored fields:
    - mask_binary
    - contour
    """
    numpy_bundle = {
        "mask_binary": _as_scalar_field_array(result["mask_binary"].astype(np.float64)),
        "contour": _as_scalar_field_array(result["contour"].astype(np.float64)),
    }
    packed = pack_numpy_fields_to_mugrid(numpy_bundle, ghosts=ghosts, verbose=verbose)
    packed["meta"] = result["params"]
    return packed


def print_field_summary(field_bundle: dict[str, Any]) -> None:
    """
    Function that prints a summary of the stored muGrid fields.
    """
    print("\nmuGrid field summary")
    print("-" * 72)

    for name, report in field_bundle["report"].items():
        print(f"{name}:")
        print(f"  status  : {report['status']}")
        print(f"  shape   : {report['shape']}")
        print(f"  p_shape : {report['field_p_shape']}")
        print(f"  pg_shape: {report['field_pg_shape']}")
        print()


# ============================================================
# Saving helpers
# ============================================================


def save_result_csvs(
    result: dict[str, Any],
    output_dir: str | os.PathLike[str],
) -> dict[str, str]:
    """
    Function that saves selected outputs as CSV files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "mask_binary_csv": str(output_dir / "grain_boundary_mask_1024x1024.csv"),
        "contour_csv": str(output_dir / "grain_boundary_contour_1024x1024.csv"),
    }

    pd.DataFrame(result["mask_binary"]).to_csv(paths["mask_binary_csv"], index=False, header=False)
    pd.DataFrame(result["contour"]).to_csv(paths["contour_csv"], index=False, header=False)

    return paths


# ============================================================
# High-level convenience API
# ============================================================


def process_npy_to_mugrid(
    input_file: str | os.PathLike[str],
    ghosts: int = 1,
    store_intermediate: bool = True,
    verbose: bool = False,
    **pipeline_kwargs,
) -> dict[str, Any]:
    """
    Function that loads a .npy image, runs the Otsu/contour pipeline,
    and packs the result into muGrid.
    """
    data = load_npy_image(input_file)
    result = run_otsu_contour_pipeline(data, **pipeline_kwargs)
    packed = pack_otsu_results_to_mugrid(
        result,
        ghosts=ghosts,
        store_intermediate=store_intermediate,
        verbose=verbose,
    )
    packed["source"] = str(input_file)
    return packed