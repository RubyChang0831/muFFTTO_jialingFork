import numpy as np
import pytest
from pathlib import Path

from muFFTTO.otsu import (
    load_npy_image,
    normalize_to_u8,
    gaussian_blur_u8,
    otsu_threshold_dark_foreground,
    clean_binary_mask,
    find_external_contour_mask,
    run_otsu_contour_pipeline,
    make_mugrid_decomposition,
    _as_scalar_field_array,
    _copy_array_into_field_p,
    make_numpy_field_bundle,
    pack_numpy_fields_to_mugrid,
    pack_otsu_results_to_mugrid,
    pack_useful_fields_to_mugrid,
    save_result_csvs,
    process_npy_to_mugrid,
)


ATOL_STRICT = 1e-12
ATOL_LOOSE = 1e-11


def make_simple_test_image(nx=16, ny=16):
    data = np.ones((nx, ny), dtype=np.float32)
    data[5:11, 7:9] = 0.0
    return data


def test_load_npy_image_loads_2d_array(tmp_path):
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    file_path = tmp_path / "img.npy"
    np.save(file_path, arr)

    loaded = load_npy_image(file_path)

    assert loaded.shape == (3, 4)
    assert loaded.dtype == np.float32
    assert np.allclose(loaded, arr, atol=ATOL_STRICT)


def test_load_npy_image_raises_for_non_2d(tmp_path):
    arr = np.zeros((2, 3, 4), dtype=np.float32)
    file_path = tmp_path / "bad.npy"
    np.save(file_path, arr)

    with pytest.raises(ValueError):
        load_npy_image(file_path)


def test_normalize_to_u8_returns_expected_ranges():
    data = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)

    img_norm, img_u8 = normalize_to_u8(data)

    assert img_norm.shape == data.shape
    assert img_u8.shape == data.shape
    assert img_norm.dtype == np.float32
    assert img_u8.dtype == np.uint8
    assert np.isclose(img_norm.min(), 0.0, atol=ATOL_STRICT)
    assert np.isclose(img_norm.max(), 1.0, atol=ATOL_STRICT)
    assert img_u8.min() == 0
    assert img_u8.max() == 255


def test_gaussian_blur_u8_preserves_shape_and_dtype():
    img_u8 = np.zeros((9, 9), dtype=np.uint8)
    img_u8[4, 4] = 255

    blur = gaussian_blur_u8(img_u8, ksize=3, sigma=1.0)

    assert blur.shape == img_u8.shape
    assert blur.dtype == np.uint8


def test_gaussian_blur_u8_raises_for_even_ksize():
    img_u8 = np.zeros((5, 5), dtype=np.uint8)

    with pytest.raises(ValueError):
        gaussian_blur_u8(img_u8, ksize=4, sigma=1.0)


def test_otsu_threshold_dark_foreground_returns_binary_mask():
    img_u8 = np.full((16, 16), 255, dtype=np.uint8)
    img_u8[6:10, 6:10] = 0

    thresh, mask_raw = otsu_threshold_dark_foreground(img_u8)

    assert isinstance(thresh, float)
    assert mask_raw.shape == img_u8.shape
    assert mask_raw.dtype == np.uint8
    assert set(np.unique(mask_raw)).issubset({0, 255})
    assert np.count_nonzero(mask_raw) > 0


def test_clean_binary_mask_returns_expected_keys_and_binary_values():
    mask_raw = np.zeros((16, 16), dtype=np.uint8)
    mask_raw[5:11, 5:11] = 255

    cleaned = clean_binary_mask(mask_raw, kernel_size=3, open_iterations=1, close_iterations=1)

    expected_keys = {"mask_open", "mask_clean", "mask_binary"}
    assert set(cleaned.keys()) == expected_keys
    assert cleaned["mask_open"].shape == mask_raw.shape
    assert cleaned["mask_clean"].shape == mask_raw.shape
    assert cleaned["mask_binary"].shape == mask_raw.shape
    assert cleaned["mask_binary"].dtype == np.uint8
    assert set(np.unique(cleaned["mask_binary"])).issubset({0, 1})


def test_find_external_contour_mask_returns_nonzero_contour_for_simple_blob():
    mask_binary = np.zeros((16, 16), dtype=np.uint8)
    mask_binary[4:12, 5:11] = 1

    contour_info = find_external_contour_mask(mask_binary)

    expected_keys = {
        "contours",
        "hierarchy",
        "contour_mask",
        "contour_pixel_count",
        "contour_count",
    }
    assert set(contour_info.keys()) == expected_keys
    assert contour_info["contour_mask"].shape == mask_binary.shape
    assert contour_info["contour_mask"].dtype == np.uint8
    assert set(np.unique(contour_info["contour_mask"])).issubset({0, 1})
    assert contour_info["contour_pixel_count"] > 0
    assert contour_info["contour_count"] >= 1


def test_run_otsu_contour_pipeline_returns_expected_keys():
    data = make_simple_test_image()

    result = run_otsu_contour_pipeline(data)

    expected_keys = {
        "image_raw",
        "image_norm",
        "image_u8",
        "image_blur",
        "mask_raw",
        "mask_open",
        "mask_clean",
        "mask_binary",
        "contour",
        "params",
    }
    assert set(result.keys()) == expected_keys
    assert result["image_raw"].shape == data.shape
    assert result["mask_binary"].shape == data.shape
    assert result["contour"].shape == data.shape
    assert result["params"]["shape"] == data.shape


def test_run_otsu_contour_pipeline_raises_for_non_2d():
    data = np.zeros((3, 4, 5), dtype=np.float32)

    with pytest.raises(ValueError):
        run_otsu_contour_pipeline(data)


def test_make_mugrid_decomposition_creates_scalar_field_with_expected_shapes():
    _, decomp = make_mugrid_decomposition(nx=8, ny=10, ghosts=1)
    field = decomp.real_field("phi", components=(1,))

    assert np.asarray(field.p).shape == (1, 8, 10)
    assert np.asarray(field.pg).shape == (1, 10, 12)


def test_as_scalar_field_array_adds_component_axis():
    arr = np.arange(12, dtype=float).reshape(3, 4)

    arr3 = _as_scalar_field_array(arr)

    assert arr3.shape == (1, 3, 4)
    assert np.allclose(arr3[0], arr, atol=ATOL_STRICT)


def test_as_scalar_field_array_raises_for_non_2d():
    arr = np.zeros((2, 3, 4), dtype=float)

    with pytest.raises(ValueError):
        _as_scalar_field_array(arr)


def test_copy_array_into_field_p_copies_exact_values():
    _, decomp = make_mugrid_decomposition(nx=4, ny=5, ghosts=1)
    field = decomp.real_field("phi", components=(1,))
    arr = np.arange(1 * 4 * 5, dtype=float).reshape(1, 4, 5)

    _copy_array_into_field_p(field, arr)

    stored = np.asarray(field.p)
    assert np.allclose(stored, arr, atol=ATOL_STRICT)


def test_copy_array_into_field_p_raises_on_shape_mismatch():
    _, decomp = make_mugrid_decomposition(nx=4, ny=5, ghosts=1)
    field = decomp.real_field("phi", components=(1,))
    arr = np.zeros((4, 5), dtype=float)

    with pytest.raises(ValueError):
        _copy_array_into_field_p(field, arr)


def test_make_numpy_field_bundle_contains_expected_keys_with_intermediate():
    data = make_simple_test_image()
    result = run_otsu_contour_pipeline(data)

    bundle = make_numpy_field_bundle(result, store_intermediate=True)

    expected_keys = {
        "image_raw",
        "image_norm",
        "image_u8",
        "image_blur",
        "mask_raw",
        "mask_open",
        "mask_clean",
        "mask_binary",
        "contour",
    }
    assert set(bundle.keys()) == expected_keys

    for arr in bundle.values():
        assert arr.shape == (1, 16, 16)


def test_make_numpy_field_bundle_without_intermediate_contains_useful_only():
    data = make_simple_test_image()
    result = run_otsu_contour_pipeline(data)

    bundle = make_numpy_field_bundle(result, store_intermediate=False)

    assert set(bundle.keys()) == {"mask_binary", "contour"}
    assert bundle["mask_binary"].shape == (1, 16, 16)
    assert bundle["contour"].shape == (1, 16, 16)


def test_pack_numpy_fields_to_mugrid_writes_all_fields():
    numpy_bundle = {
        "a": np.ones((1, 6, 7), dtype=float),
        "b": np.zeros((1, 6, 7), dtype=float),
    }

    packed = pack_numpy_fields_to_mugrid(numpy_bundle, ghosts=1, verbose=False)

    assert set(packed["fields"].keys()) == {"a", "b"}
    assert set(packed["report"].keys()) == {"a", "b"}
    assert np.allclose(np.asarray(packed["fields"]["a"].p), numpy_bundle["a"], atol=ATOL_STRICT)
    assert np.allclose(np.asarray(packed["fields"]["b"].p), numpy_bundle["b"], atol=ATOL_STRICT)


def test_pack_otsu_results_to_mugrid_returns_meta_and_fields():
    data = make_simple_test_image()
    result = run_otsu_contour_pipeline(data)

    packed = pack_otsu_results_to_mugrid(
        result,
        ghosts=1,
        store_intermediate=False,
        verbose=False,
    )

    assert "meta" in packed
    assert set(packed["fields"].keys()) == {"mask_binary", "contour"}
    assert packed["meta"]["shape"] == data.shape


def test_pack_useful_fields_to_mugrid_returns_only_selected_fields():
    data = make_simple_test_image()
    result = run_otsu_contour_pipeline(data)

    packed = pack_useful_fields_to_mugrid(result, ghosts=1, verbose=False)

    assert set(packed["fields"].keys()) == {"mask_binary", "contour"}
    assert set(packed["numpy"].keys()) == {"mask_binary", "contour"}
    assert set(packed["report"].keys()) == {"mask_binary", "contour"}


def test_save_result_csvs_creates_output_files(tmp_path):
    data = make_simple_test_image()
    result = run_otsu_contour_pipeline(data)

    paths = save_result_csvs(result, tmp_path)

    assert "mask_binary_csv" in paths
    assert "contour_csv" in paths
    assert Path(paths["mask_binary_csv"]).exists()
    assert Path(paths["contour_csv"]).exists()


def test_process_npy_to_mugrid_runs_end_to_end(tmp_path):
    data = make_simple_test_image(12, 14)
    input_file = tmp_path / "input.npy"
    np.save(input_file, data)

    packed = process_npy_to_mugrid(
        input_file,
        ghosts=1,
        store_intermediate=False,
        verbose=False,
    )

    assert packed["source"] == str(input_file)
    assert set(packed["fields"].keys()) == {"mask_binary", "contour"}
    assert packed["meta"]["shape"] == (12, 14)