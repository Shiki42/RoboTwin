from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import zlib

import numpy as np

ROBOTWIN_POLICY_PROTOCOL = "robotwin_pi0_v3_zlib_images"
_IMAGE_ENCODING = "zlib_raw_uint8_v1"
_MAX_IMAGE_BYTES = 64 * 1024 * 1024


def encode_images(images: Sequence[np.ndarray]) -> list[dict[str, object]]:
    """Losslessly encode the three current RoboTwin RGB observations."""
    if len(images) != 3:
        raise ValueError(f"expected exactly three RoboTwin images, got {len(images)}")

    encoded = []
    for index, image in enumerate(images):
        array = np.asarray(image)
        if array.dtype != np.uint8:
            raise ValueError(
                f"RoboTwin image {index} must be uint8, got {array.dtype}"
            )
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(
                f"RoboTwin image {index} must be HWC RGB, got {array.shape}"
            )
        contiguous = np.ascontiguousarray(array)
        if contiguous.nbytes > _MAX_IMAGE_BYTES:
            raise ValueError(
                f"RoboTwin image {index} is too large: {contiguous.nbytes} bytes"
            )
        encoded.append(
            {
                "encoding": _IMAGE_ENCODING,
                "shape": list(contiguous.shape),
                "data": zlib.compress(
                    contiguous.tobytes(order="C"),
                    level=zlib.Z_BEST_SPEED,
                ),
            }
        )
    return encoded


def decode_images(payload: object) -> list[np.ndarray]:
    """Decode lossless image envelopes and reject malformed transport input."""
    if not isinstance(payload, list) or len(payload) != 3:
        raise ValueError("compressed RoboTwin observation must contain three images")

    decoded = []
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise ValueError(f"compressed RoboTwin image {index} must be a mapping")
        if set(item) != {"encoding", "shape", "data"}:
            raise ValueError(f"compressed RoboTwin image {index} has invalid fields")
        if item["encoding"] != _IMAGE_ENCODING:
            raise ValueError(
                f"compressed RoboTwin image {index} has unknown encoding"
            )
        shape = item["shape"]
        valid_shape = (
            isinstance(shape, list)
            and len(shape) == 3
            and all(
                not isinstance(value, bool) and isinstance(value, int) and value > 0
                for value in shape
            )
            and shape[2] == 3
        )
        if not valid_shape:
            raise ValueError(f"compressed RoboTwin image {index} has invalid shape")
        expected_bytes = math.prod(shape)
        if expected_bytes > _MAX_IMAGE_BYTES:
            raise ValueError(f"compressed RoboTwin image {index} is too large")
        data = item["data"]
        if not isinstance(data, bytes | bytearray | memoryview):
            raise ValueError(f"compressed RoboTwin image {index} has invalid data")
        try:
            raw = zlib.decompress(data)
        except zlib.error as error:
            raise ValueError(
                f"compressed RoboTwin image {index} has invalid zlib data"
            ) from error
        if len(raw) != expected_bytes:
            raise ValueError(
                f"compressed RoboTwin image {index} byte count mismatch"
            )
        decoded.append(
            np.frombuffer(raw, dtype=np.uint8).reshape(tuple(shape)).copy()
        )
    return decoded
