from __future__ import annotations

from typing import Any

import cv2
import numpy as np


# description:

def _normalize_to_u8(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize a 2D image to [0, 1] and uint8 [0, 255]."""
    data = np.asarray(data, dtype=np.float32)

    if data.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {data.shape}.")

    data_min = float(data.min())
    data_max = float(data.max())
    value_range = data_max - data_min

    if value_range <= np.finfo(np.float32).eps:
        image_norm = np.zeros_like(data, dtype=np.float32)
    else:
        image_norm = (data - data_min) / value_range

    image_u8 = np.clip(image_norm * 255.0, 0, 255).astype(np.uint8)
    return image_norm, image_u8


def _validate_odd_positive(name: str, value: int) -> None:
    """Validate a positive odd OpenCV kernel size."""
    if value <= 0 or value % 2 == 0:
        raise ValueError(f"{name} must be a positive odd integer, got {value}.")


def otsu_edgeDetection_and_phaseIndicator(
        data: np.ndarray,
        blur_ksize: int = 7,
        blur_sigma: float = 1.5,
        morph_kernel_size: int = 3,
        open_iterations: int = 1,
        close_iterations: int = 1,
        regions_label: bool = True,
        connectivity: int = 8,
        return_intermediate: bool = False,

) -> dict[str, Any]:
    """
    Segment dark regions in a two-dimensional grayscale image, extract their
    external boundaries, and optionally assign a unique integer label to each
    disconnected foreground region.

    The input image is normalized to uint8 intensity values, smoothed with a
    Gaussian filter, and segmented using inverted Otsu thresholding. Morphological
    opening and closing are then applied to reduce isolated noise, remove small
    foreground artifacts, fill small holes, and close short gaps.

    Inverted Otsu thresholding treats dark pixels as foreground:

    - ``phase_mask_binary == 1`` identifies dark segmented regions.
    - ``phase_mask_binary == 0`` identifies bright/background regions.

    When ``regions_label=True``, connected-component analysis assigns one unique
    integer ID to each disconnected foreground region. Background pixels always
    have label ``0``; foreground region labels are consecutive integers
    ``1, 2, ..., N``.

    This function currently distinguishes regions by geometric connectivity only.
    Therefore, two disconnected regions receive different labels even when their
    average grayscale values are similar. A later processing step may group such
    regions into shared material classes based on their mean grayscale values.

    Parameters
    ----------
    data : numpy.ndarray
        Two-dimensional grayscale input image with shape ``(nx, ny)``. Integer and
        floating-point input dtypes are supported.

    blur_ksize : int, default=7
        Positive odd Gaussian-kernel width and height. Larger values produce
        stronger smoothing before Otsu thresholding.

    blur_sigma : float, default=1.5
        Standard deviation of the Gaussian filter. Must be non-negative.

    morph_kernel_size : int, default=3
        Positive odd side length of the square structuring element used for
        morphological opening and closing.

    open_iterations : int, default=1
        Number of morphological-opening iterations. Opening suppresses small
        isolated foreground pixels and thin noise.

    close_iterations : int, default=1
        Number of morphological-closing iterations. Closing fills small holes and
        connects short gaps inside foreground regions.

    regions_label : bool, default=True
        If ``True``, perform connected-component labeling on the cleaned binary
        foreground mask.

        - ``True``: ``phase_mask_label`` has dtype ``int32`` and contains labels
          ``0, 1, ..., N``.
        - ``False``: ``phase_mask_label`` is a uint8 copy of
          ``phase_mask_binary`` and contains only ``0`` and ``1``.

    connectivity : int, default=8
        Connectivity used for connected-component labeling when
        ``regions_label=True``.

        - ``4`` connects pixels sharing a horizontal or vertical edge.
        - ``8`` additionally connects pixels touching diagonally at a corner.

    return_intermediate : bool, default=False
        If ``True``, include normalized, blurred, thresholded, and morphology
        intermediate arrays in the returned dictionary.

    Returns
    -------
    dict[str, Any]
        Dictionary containing the following arrays with shape ``(nx, ny)``:

        ``edge_mask`` : numpy.ndarray, uint8
            One-pixel-wide mask of external contours. A value of ``1`` marks an
            outer boundary of a segmented foreground region.

        ``phase_mask_binary`` : numpy.ndarray, uint8
            Cleaned binary segmentation mask with values ``0`` for background and
            ``1`` for dark foreground regions.

        ``phase_mask_label`` : numpy.ndarray
            Region-label field.

            - If ``regions_label=True``, dtype is ``int32`` and values are
              ``0, 1, ..., N``.
            - If ``regions_label=False``, dtype is ``uint8`` and values are
              ``0`` and ``1``.

        When ``regions_label=True``, the dictionary additionally contains:

        ``number_of_phase_regions`` : int
            Number of disconnected foreground regions. Background label ``0`` is
            excluded.

        ``phase_region_stats`` : numpy.ndarray
            OpenCV connected-component statistics. Each row corresponds to one
            region label, including background label ``0``. The columns contain
            bounding-box x/y coordinates, width, height, and pixel area.

        ``phase_region_centroids`` : numpy.ndarray
            Centroid coordinates ``(x, y)`` for every label, including background.

        When ``return_intermediate=True``, the dictionary additionally contains:

        ``image_norm`` : numpy.ndarray
            Input image normalized to the interval ``[0, 1]``.

        ``image_u8`` : numpy.ndarray
            Normalized uint8 image with intensity values in ``[0, 255]``.

        ``image_blur`` : numpy.ndarray
            Gaussian-smoothed uint8 image.

        ``mask_raw`` : numpy.ndarray, uint8
            Binary result directly after inverted Otsu thresholding.

        ``mask_open`` : numpy.ndarray, uint8
            Binary result after morphological opening.

        ``mask_clean`` : numpy.ndarray, uint8
            Binary result after opening and closing; equal to
            ``phase_mask_binary``.

        ``otsu_threshold`` : float
            Automatically selected Otsu threshold in uint8 intensity units.

        ``contour_count`` : int
            Number of external contours detected from ``phase_mask_binary``.

    Notes
    -----
    The function applies standard non-periodic image connectivity. Hence, a
    physical region crossing a periodic simulation boundary may be assigned
    separate labels at opposite image edges. Periodic label merging should be
    implemented before using region labels for periodic material identification or
    periodic adaptive-grid decisions.

    The returned ``phase_mask_label`` describes disconnected geometric regions,
    not material classes. In a future extension, regions with similar mean
    grayscale values can be grouped into a separate ``material_label`` field,
    while retaining ``phase_mask_label`` for region geometry.
    """

    _validate_odd_positive("blur_ksize", blur_ksize)
    _validate_odd_positive("morph_kernel_size", morph_kernel_size)

    if blur_sigma < 0:
        raise ValueError(f"blur_sigma must be non-negative, got {blur_sigma}.")
    if open_iterations < 0:
        raise ValueError(
            f"open_iterations must be non-negative, got {open_iterations}."
        )
    if close_iterations < 0:
        raise ValueError(
            f"close_iterations must be non-negative, got {close_iterations}."
        )
    if connectivity not in (4, 8):
        raise ValueError(
            f"connectivity must be 4 or 8, got {connectivity}."
        )

    image_norm, image_u8 = _normalize_to_u8(data)

    image_blur = cv2.GaussianBlur(
        image_u8,
        (blur_ksize, blur_ksize),
        blur_sigma,
    )

    otsu_threshold, mask_raw_u8 = cv2.threshold(
        image_blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    kernel = np.ones(
        (morph_kernel_size, morph_kernel_size),
        dtype=np.uint8,
    )

    mask_open_u8 = cv2.morphologyEx(
        mask_raw_u8,
        cv2.MORPH_OPEN,
        kernel,
        iterations=open_iterations,
    )

    mask_clean_u8 = cv2.morphologyEx(
        mask_open_u8,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=close_iterations,
    )

    # Binary phase indicator: values are always 0 or 1.
    phase_mask_binary = (mask_clean_u8 > 0).astype(np.uint8)

    # Detect edges from the binary phase mask.
    contours, _ = cv2.findContours(
        (phase_mask_binary * 255).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    edge_mask = np.zeros_like(phase_mask_binary, dtype=np.uint8)

    cv2.drawContours(
        edge_mask,
        contours,
        contourIdx=-1,
        color=1,
        thickness=1,
    )

    # Default behavior: label field is the same as the binary mask.
    phase_mask_label = phase_mask_binary.copy()

    # Optional: assign one integer label to each connected region.
    if regions_label:
        number_of_labels, phase_mask_label, phase_region_stats, phase_region_centroids = (
            cv2.connectedComponentsWithStats(
                phase_mask_binary,
                connectivity=connectivity,
                ltype=cv2.CV_32S,
            )
        )

    results: dict[str, Any] = {
        "edge_mask": edge_mask,
        "phase_mask_binary": phase_mask_binary,
        "phase_mask_label": phase_mask_label,
    }

    # These data exist only when connected-component labeling was requested.
    if regions_label:
        results.update(
            {
                "number_of_phase_regions": int(number_of_labels - 1),
                "phase_region_stats": phase_region_stats,
                "phase_region_centroids": phase_region_centroids,
            }
        )

    if return_intermediate:
        results.update(
            {
                "image_norm": image_norm,
                "image_u8": image_u8,
                "image_blur": image_blur,
                "mask_raw": (mask_raw_u8 > 0).astype(np.uint8),
                "mask_open": (mask_open_u8 > 0).astype(np.uint8),
                "mask_clean": phase_mask_binary,
                "otsu_threshold": float(otsu_threshold),
                "contour_count": int(len(contours)),
            }
        )

    return results