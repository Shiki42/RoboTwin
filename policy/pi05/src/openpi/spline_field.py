"""Deterministic B-spline representation for continuous action fields."""

from __future__ import annotations

import functools

import numpy as np


def _validate_shape(control_horizon: int, control_points: int, degree: int, regularization: float) -> None:
    if control_horizon <= control_points:
        raise ValueError("control_horizon must be greater than control_points")
    if control_points <= degree:
        raise ValueError("control_points must be greater than degree")
    if degree < 1:
        raise ValueError("degree must be positive")
    if regularization < 0:
        raise ValueError("regularization must be non-negative")


def _clamped_uniform_knots(control_points: int, degree: int) -> np.ndarray:
    interior_count = control_points - degree - 1
    interior = np.arange(1, interior_count + 1, dtype=np.float64) / (interior_count + 1)
    return np.concatenate(
        [
            np.zeros(degree + 1, dtype=np.float64),
            interior,
            np.ones(degree + 1, dtype=np.float64),
        ]
    )


def evaluate_basis(sample_times: np.ndarray, control_points: int, degree: int = 3) -> np.ndarray:
    """Evaluate a clamped, uniform B-spline basis at normalized times in [0, 1]."""
    times = np.asarray(sample_times, dtype=np.float64)
    if times.ndim != 1:
        raise ValueError(f"sample_times must be one-dimensional, got {times.shape}")
    if not np.all(np.isfinite(times)):
        raise ValueError("sample_times contains non-finite values")
    if np.any(times < 0.0) or np.any(times > 1.0):
        raise ValueError("sample_times must lie in [0, 1]")
    if control_points <= degree:
        raise ValueError("control_points must be greater than degree")

    knots = _clamped_uniform_knots(control_points, degree)
    basis = np.zeros((times.size, knots.size - 1), dtype=np.float64)
    for index in range(basis.shape[1]):
        basis[:, index] = (knots[index] <= times) & (times < knots[index + 1])

    for order in range(1, degree + 1):
        next_basis = np.zeros((times.size, basis.shape[1] - 1), dtype=np.float64)
        for index in range(next_basis.shape[1]):
            left_denominator = knots[index + order] - knots[index]
            right_denominator = knots[index + order + 1] - knots[index + 1]
            if left_denominator > 0:
                next_basis[:, index] += (times - knots[index]) / left_denominator * basis[:, index]
            if right_denominator > 0:
                next_basis[:, index] += (knots[index + order + 1] - times) / right_denominator * basis[:, index + 1]
        basis = next_basis

    basis[times == 1.0] = 0.0
    basis[times == 1.0, -1] = 1.0
    return basis


@functools.cache
def spline_matrices(
    control_horizon: int,
    control_points: int,
    degree: int = 3,
    regularization: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the decode basis and regularized least-squares encoder."""
    _validate_shape(control_horizon, control_points, degree, regularization)
    times = np.linspace(0.0, 1.0, control_horizon, dtype=np.float64)
    basis = evaluate_basis(times, control_points, degree)
    second_difference = np.diff(np.eye(control_points, dtype=np.float64), n=2, axis=0)
    system = basis.T @ basis + regularization * second_difference.T @ second_difference
    projector = np.linalg.solve(system, basis.T)
    basis.setflags(write=False)
    projector.setflags(write=False)
    return basis, projector


def encode_actions(
    actions: np.ndarray,
    *,
    control_horizon: int,
    control_points: int,
    degree: int = 3,
    regularization: float = 1e-6,
) -> np.ndarray:
    """Fit spline coefficients to arrays shaped [..., control_horizon, action_dim]."""
    values = np.asarray(actions)
    if values.ndim < 2 or values.shape[-2] != control_horizon:
        raise ValueError(f"expected action horizon {control_horizon}, got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("actions contain non-finite values")
    _, projector = spline_matrices(control_horizon, control_points, degree, regularization)
    coefficients = np.einsum("mn,...nd->...md", projector, values, optimize=True)
    return coefficients.astype(values.dtype, copy=False)


def decode_actions(
    coefficients: np.ndarray,
    *,
    control_horizon: int,
    control_points: int,
    degree: int = 3,
    regularization: float = 1e-6,
) -> np.ndarray:
    """Decode arrays shaped [..., control_points, action_dim] on the training time grid."""
    values = np.asarray(coefficients)
    if values.ndim < 2 or values.shape[-2] != control_points:
        raise ValueError(f"expected {control_points} spline control points, got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("spline coefficients contain non-finite values")
    basis, _ = spline_matrices(control_horizon, control_points, degree, regularization)
    actions = np.einsum("nm,...md->...nd", basis, values, optimize=True)
    return actions.astype(values.dtype, copy=False)


def evaluate_actions(
    coefficients: np.ndarray,
    sample_times: np.ndarray,
    *,
    degree: int = 3,
) -> np.ndarray:
    """Query a spline action field at arbitrary normalized times."""
    values = np.asarray(coefficients)
    if values.ndim < 2:
        raise ValueError(f"expected coefficient array [..., control_points, action_dim], got {values.shape}")
    basis = evaluate_basis(sample_times, values.shape[-2], degree)
    actions = np.einsum("nm,...md->...nd", basis, values, optimize=True)
    return actions.astype(values.dtype, copy=False)
