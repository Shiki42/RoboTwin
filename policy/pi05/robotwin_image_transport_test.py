import numpy as np
import pytest

from robotwin_image_transport import decode_images
from robotwin_image_transport import encode_images


def test_image_transport_is_lossless_for_noncontiguous_rgb():
    base = np.zeros((16, 24, 3), dtype=np.uint8)
    base[..., 0] = np.arange(24, dtype=np.uint8)
    base[..., 1] = np.arange(16, dtype=np.uint8)[:, None]
    images = [base, base[:, ::-1], np.rot90(base)]

    encoded = encode_images(images)
    decoded = decode_images(encoded)

    assert sum(len(item["data"]) for item in encoded) < sum(
        image.nbytes for image in images
    )
    assert all(array.flags.c_contiguous for array in decoded)
    assert all(
        np.array_equal(expected, actual)
        for expected, actual in zip(images, decoded, strict=True)
    )


def test_image_transport_rejects_non_uint8_and_wrong_count():
    with pytest.raises(ValueError, match="exactly three"):
        encode_images([np.zeros((2, 2, 3), dtype=np.uint8)])
    with pytest.raises(ValueError, match="must be uint8"):
        encode_images([np.zeros((2, 2, 3), dtype=np.float32)] * 3)


def test_image_transport_rejects_corrupt_payload():
    encoded = encode_images([np.zeros((2, 2, 3), dtype=np.uint8)] * 3)
    encoded[1] = {**encoded[1], "data": b"not-zlib"}

    with pytest.raises(ValueError, match="invalid zlib data"):
        decode_images(encoded)


def test_image_transport_rejects_shape_byte_mismatch():
    encoded = encode_images([np.zeros((2, 2, 3), dtype=np.uint8)] * 3)
    encoded[0] = {**encoded[0], "shape": [3, 3, 3]}

    with pytest.raises(ValueError, match="byte count mismatch"):
        decode_images(encoded)
