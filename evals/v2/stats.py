"""Small statistics helpers shared by the v2 evals."""

import math


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for ``k`` successes in ``n`` trials (default 95%).

    Raises:
        ValueError: if ``n <= 0``, ``k < 0`` or ``k > n``.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if k < 0 or k > n:
        raise ValueError(f"k must be in [0, n], got k={k}, n={n}")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)
